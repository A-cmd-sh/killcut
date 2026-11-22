#!/usr/bin/env python3
"""
Valorant キル集 自動モンタージュ作成ツール

使い方:
    python montage.py <動画のフォルダ> [--name プレイヤー名] [オプション]

■ キルログモード (--name を指定)
  キルフィード (画面右上のキルログ) を文字認識して、指定した名前のプレイヤーが
  キルした場面だけを切り出す。フル試合の録画をそのまま入れてもOK。
  連続キルは1つのクリップにまとめる。名前は複数指定可 (--name A --name B)。

■ 音量モード (--name なし)
  各クリップから「音が一番激しい区間」(銃声・キル音) を切り出す。
  Medal / Outplayed などで保存した短いキルクリップ向け。

■ 音楽
  フォルダに曲 (mp3/wav/m4a/flac/ogg) を入れておくと、名前順につないでBGMにし、
  ビートに合わせてカットを切り替える。--music で別の場所の曲を複数指定してもよい。

出力: <フォルダ>/montage.mp4

必要なもの: ffmpeg (PATHに通っていること), Python 3.9+, numpy
  ビート同期: pip install librosa   (無ければ --bpm で指定)
  キルログモード: pip install easyocr
"""
import argparse
import difflib
import json
import random
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
CACHE_VERSION = 1


# ================================================================ ffmpeg helpers
def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode(errors="replace")[-2000:])
        raise RuntimeError("ffmpeg failed: " + " ".join(map(str, cmd[:6])) + " ...")
    return r.stdout


def probe(path):
    out = run(["ffprobe", "-v", "error", "-show_entries",
               "format=duration:stream=codec_type,width,height", "-of", "json", str(path)])
    info = json.loads(out)
    dur = float(info["format"]["duration"])
    streams = info.get("streams", [])
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    return dur, has_audio, int(v.get("width", 1920)), int(v.get("height", 1080))


def load_audio_mono(path, sr=8000):
    raw = run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
               "-f", "s16le", "-"])
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


# ================================================================ 音量モード
def best_window(path, dur, has_audio, length, peak_pos):
    """音のエネルギーが最大の瞬間 (キル音・銃声) が窓の peak_pos の位置に来る開始時刻を返す"""
    if length >= dur:
        return 0.0
    if not has_audio:
        return max(0.0, dur - length - 0.5)
    y, sr = load_audio_mono(path)
    hop = int(sr * 0.05)
    n = len(y) // hop
    if n < 2:
        return max(0.0, dur - length)
    rms = np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    smooth = np.convolve(rms, np.ones(10) / 10, mode="same")
    if n > 12:
        smooth[:6] = 0
        smooth[-6:] = 0
    peak_t = float(np.argmax(smooth)) * 0.05
    return float(min(max(0.0, peak_t - length * peak_pos), dur - length))


# ================================================================ キルログモード
def norm_name(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return "".join(ch for ch in s if ch.isalnum())


def name_score(text, target):
    """OCR文字列 text が名前 target にどれだけ近いか (0-1)"""
    t, n = norm_name(text), norm_name(target)
    if not t or not n:
        return 0.0
    if n in t:
        return 1.0
    best = difflib.SequenceMatcher(None, t, n).ratio()
    # OCRで前後に余計な文字が付いた場合に備えて部分一致も見る
    for L in (len(n) - 1, len(n), len(n) + 1):
        if 0 < L < len(t):
            for i in range(len(t) - L + 1):
                best = max(best, difflib.SequenceMatcher(None, t[i:i + L], n).ratio())
    return best


_reader = None


def get_reader(gpu):
    global _reader
    if _reader is None:
        try:
            import easyocr
        except ImportError:
            sys.exit("キルログモードには easyocr が必要です:  pip install easyocr")
        print("  文字認識モデルを読み込み中 (初回はダウンロードに数分かかります)...")
        _reader = easyocr.Reader(["ja", "en"], gpu=gpu, verbose=False)
    return _reader


def parse_region(s):
    vals = [float(v) for v in s.split(",")]
    if len(vals) != 4:
        raise ValueError("--feed-region は x0,y0,x1,y1 (画面比 0-1) で指定")
    return vals


def read_feed_rows(frame, reader, scale):
    """キルフィード画像から [(行のy, [(x0, x1, text, conf), ...左から順]), ...] を返す"""
    res = reader.readtext(frame, detail=1, paragraph=False)
    boxes = []
    for bbox, text, conf in res:
        if conf < 0.15 or not norm_name(text):
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        boxes.append(dict(x0=min(xs) / scale, x1=max(xs) / scale, yc=(min(ys) + max(ys)) / 2 / scale,
                          h=(max(ys) - min(ys)) / scale, text=text, conf=float(conf)))
    boxes.sort(key=lambda b: b["yc"])
    rows = []
    for b in boxes:
        if rows and abs(rows[-1]["yc"] - b["yc"]) < max(6.0, 0.6 * b["h"]):
            rows[-1]["boxes"].append(b)
        else:
            rows.append(dict(yc=b["yc"], boxes=[b]))
    for r in rows:
        r["boxes"].sort(key=lambda b: b["x0"])
    return rows


def find_kills_in_rows(rows, names, crop_w, thr):
    """名前がキラー側 (左) にいる行を探して [(犠牲者テキスト), ...] を返す"""
    found = []
    for r in rows:
        bs = r["boxes"]
        for i, b in enumerate(bs):
            if max(name_score(b["text"], n) for n in names) < thr:
                continue
            right = bs[i + 1:]
            if right:
                # 自分の右に別の名前がある → 自分がキラー
                found.append(right[-1]["text"])
            elif len(bs) == 1 and b["x1"] < crop_w * 0.85:
                # 相手の名前が読めなかった場合: 右端に寄っていなければキラー側とみなす
                found.append("?")
            # それ以外 (自分が右端) はデス
            break
    return found


def detect_kills(clip, dur, vw, vh, names, args, cache_dir, debug_dir):
    """クリップ内で指定プレイヤーがキルした時刻 (秒) のリストを返す (結果はキャッシュ)"""
    key = dict(v=CACHE_VERSION, names=sorted(names), region=args.feed_region, sample=args.sample, ocr=args.ocr_scale,
               thr=args.match, size=clip.stat().st_size, mtime=int(clip.stat().st_mtime))
    cache = cache_dir / (clip.name + ".kills.json")
    if cache.exists() and not debug_dir:
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("key") == key:
                return data["kills"]
        except Exception:
            pass

    reader = get_reader(not args.cpu)
    x0, y0, x1, y1 = parse_region(args.feed_region)
    cx, cy = int(vw * x0) // 2 * 2, int(vh * y0) // 2 * 2
    cw, ch = int(vw * (x1 - x0)) // 2 * 2, int(vh * (y1 - y0)) // 2 * 2
    # 小さい文字は拡大した方が認識しやすい (1080p基準で2倍相当)
    scale = max(1.0, args.ocr_scale * 1080 / vh)
    ow, oh = int(cw * scale) // 2 * 2, int(ch * scale) // 2 * 2
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(clip), "-vf",
         f"fps=1/{args.sample},crop={cw}:{ch}:{cx}:{cy},scale={ow}:{oh}:flags=lanczos",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fsize = ow * oh * 3
    total = int(dur / args.sample) + 1
    kills, last_seen = [], {}
    i = 0
    while True:
        buf = proc.stdout.read(fsize)
        if len(buf) < fsize:
            break
        t = i * args.sample
        frame = np.frombuffer(buf, np.uint8).reshape(oh, ow, 3)
        rows = read_feed_rows(frame, reader, scale)
        victims = find_kills_in_rows(rows, names, cw, args.match)
        for v in victims:
            k = norm_name(v) or "?"
            # 同じキルはキルログに数秒残るので、一定時間内の再検出は無視
            # 直近に見た同じ犠牲者 (OCRの揺れを許容) がいれば同じキル
            recent = [kk for kk, lt in last_seen.items() if t - lt < args.feed_hold and
                      (kk == k or (k != "?" and kk != "?" and
                                   difflib.SequenceMatcher(None, kk, k).ratio() >= 0.75))]
            if k == "?" and len(last_seen) and any(t - lt < args.feed_hold for lt in last_seen.values()):
                recent = recent or ["?"]  # 名前が読めなかった行は、他のキルが表示中なら同じものとみなす
            if recent:
                for kk in recent:
                    last_seen[kk] = t
                continue
            last_seen[k] = t
            kills.append(round(max(0.0, t - args.sample / 2), 2))
            print(f"    キル検出 {t:7.1f}s  → {v}")
        if debug_dir and i < 400:
            save_debug(frame, rows, debug_dir / f"{clip.stem}_{t:07.1f}.png", scale)
        i += 1
        if i % 20 == 0:
            print(f"    解析中 {min(100, i * 100 // total):3d}%", end="\r")
    proc.wait()
    print(" " * 30, end="\r")
    cache_dir.mkdir(exist_ok=True)
    cache.write_text(json.dumps(dict(key=key, kills=kills), ensure_ascii=False), encoding="utf-8")
    return kills


def save_debug(frame, rows, path, scale):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    path.parent.mkdir(exist_ok=True)
    img = Image.fromarray(frame.copy())
    d = ImageDraw.Draw(img)
    for r in rows:
        for b in r["boxes"]:
            d.rectangle([b["x0"] * scale, (b["yc"] - b["h"] / 2) * scale,
                         b["x1"] * scale, (b["yc"] + b["h"] / 2) * scale], outline=(255, 0, 255), width=2)
    img.save(path)


def kills_to_segments(kills, dur, pre, post, merge_gap):
    """キル時刻 → [(開始, 終了, キル数)]  連続キルはまとめる"""
    segs = []
    for t in sorted(kills):
        s, e = max(0.0, t - pre), min(dur, t + post)
        if segs and s - segs[-1][1] <= merge_gap:
            segs[-1] = (segs[-1][0], e, segs[-1][2] + 1)
        else:
            segs.append((s, e, 1))
    return segs


# ================================================================ モザイク
def mosaic_filter(regions, vw, vh, style, strength, tail):
    """指定範囲にモザイク/ぼかしをかけてから出力サイズに整える filter_complex を返す"""
    if not regions:
        return f"[0:v:0]{tail}[vout]"
    n = len(regions)
    fc = f"[0:v:0]split={n + 1}[base]" + "".join(f"[s{i}]" for i in range(n)) + ";"
    cur = "base"
    for i, r in enumerate(regions):
        x0, y0, x1, y1 = parse_region(r)
        x, y = int(vw * x0) // 2 * 2, int(vh * y0) // 2 * 2
        w, h = max(2, int(vw * (x1 - x0)) // 2 * 2), max(2, int(vh * (y1 - y0)) // 2 * 2)
        if style == "blur":
            eff = f"boxblur={max(2, strength)}:2"
        else:
            sw, sh = max(1, w // strength), max(1, h // strength)
            eff = f"scale={sw}:{sh}:flags=area,scale={w}:{h}:flags=neighbor"
        fc += f"[s{i}]crop={w}:{h}:{x}:{y},{eff}[m{i}];[{cur}][m{i}]overlay={x}:{y}[o{i}];"
        cur = f"o{i}"
    return fc + f"[{cur}]{tail}[vout]"


# ================================================================ 音楽
def prepare_music(paths, tmp):
    """複数の曲を1本のwavにつなぐ"""
    if len(paths) == 1:
        return paths[0]
    out = tmp / "bgm_joined.wav"
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for p in paths:
        cmd += ["-i", str(p)]
    fc = "".join(f"[{i}:a]aresample=44100,aformat=channel_layouts=stereo[a{i}];" for i in range(len(paths)))
    fc += "".join(f"[a{i}]" for i in range(len(paths))) + f"concat=n={len(paths)}:v=0:a=1[a]"
    run(cmd + ["-filter_complex", fc, "-map", "[a]", str(out)])
    return out


def get_beats(music, bpm, music_start):
    if bpm:
        return music_start + np.arange(0, 4000) * (60.0 / bpm)
    try:
        import librosa
    except ImportError:
        print("! librosa が無いのでビート同期できません (pip install librosa か --bpm で指定)")
        return None
    print("  曲のビートを解析中...")
    y, sr = librosa.load(str(music), sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    beats = np.asarray(beats, dtype=float)
    beats = beats[beats >= music_start]
    if len(beats) < 4:
        return None
    print(f"  テンポ: 約 {float(np.atleast_1d(tempo)[0]):.0f} BPM")
    interval = float(np.median(np.diff(beats)))
    return np.concatenate([beats, beats[-1] + interval * np.arange(1, 4000)])


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description="キル動画のフォルダからモンタージュ動画を作る")
    ap.add_argument("folder", nargs="?", help="動画の入ったフォルダ (省略すると対話モード)")
    g = ap.add_argument_group("キルログモード")
    g.add_argument("--name", action="append", default=[],
                   help="キルを拾うプレイヤー名 (複数可)。指定するとキルログモードになる")
    g.add_argument("--pre", type=float, default=3.5, help="キルの何秒前から使うか")
    g.add_argument("--post", type=float, default=1.2, help="キルの何秒後まで使うか")
    g.add_argument("--merge-gap", type=float, default=1.5, help="この秒数以内に続くキルは1クリップにまとめる")
    g.add_argument("--feed-region", default="0.68,0.06,0.995,0.40",
                   help="キルログの範囲 x0,y0,x1,y1 (画面比)")
    g.add_argument("--sample", type=float, default=1.0, help="キルログを何秒ごとに読むか")
    g.add_argument("--match", type=float, default=0.75, help="名前一致の判定しきい値 (0-1)")
    g.add_argument("--feed-hold", type=float, default=7.0, help="同じキルを重複カウントしない秒数")
    g.add_argument("--cpu", action="store_true", help="GPUを使わない (デフォルトはGPUがあれば使う)")
    g.add_argument("--ocr-scale", type=float, default=1.0, help="文字が小さく読めない時は 1.5-2 に")
    g.add_argument("--debug-feed", action="store_true",
                   help="読み取ったキルログ画像を _feed_debug フォルダに保存 (範囲調整用)")
    g.add_argument("--detect-only", action="store_true", help="キル検出だけして動画は作らない")
    m = ap.add_argument_group("音楽")
    m.add_argument("--music", nargs="+", help="BGM (複数指定でつなげる)。省略時はフォルダ内の曲を使用")
    m.add_argument("--no-music", action="store_true", help="BGMを使わない")
    m.add_argument("--bpm", type=float, help="BPMを手動指定")
    m.add_argument("--music-start", type=float, default=0.0, help="曲を何秒目から使うか")
    m.add_argument("--music-vol", type=float, default=1.0)
    m.add_argument("--game-vol", type=float, default=0.35, help="BGMありの時のゲーム音量")
    m.add_argument("--max-beats", type=int, default=16, help="1クリップの最大ビート数")
    z = ap.add_argument_group("モザイク")
    z.add_argument("--mosaic", action="store_true", help="キルログ (--feed-region の範囲) にモザイクをかける")
    z.add_argument("--mosaic-style", choices=["pixel", "blur"], default="pixel", help="pixel=モザイク, blur=ぼかし")
    z.add_argument("--mosaic-region", action="append", default=[],
                   help="追加でモザイクをかける範囲 x0,y0,x1,y1 (画面比, 複数可)")
    z.add_argument("--mosaic-strength", type=int, default=16, help="モザイクの粗さ")
    o = ap.add_argument_group("その他")
    o.add_argument("--out", help="出力ファイル (デフォルト: <フォルダ>/montage.mp4)")
    o.add_argument("--beats", type=int, default=8, help="音量モードでの1クリップのビート数")
    o.add_argument("--clip-len", type=float, default=4.0, help="音量モード・BGMなしの時の1クリップの秒数")
    o.add_argument("--peak-pos", type=float, default=0.65)
    o.add_argument("--order", choices=["name", "date", "shuffle"], default="name")
    o.add_argument("--res", default="1920x1080", help="出力解像度 (縦型なら 1080x1920)")
    o.add_argument("--fps", type=int, default=60)
    o.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()
    # 引数なし or フォルダだけ (exeにフォルダをドラッグ) の場合は質問形式で設定する
    interactive = len(sys.argv) == 1 or (len(sys.argv) == 2 and not sys.argv[1].startswith("-"))
    if interactive:
        try:
            ask(args)
        except (KeyboardInterrupt, EOFError):
            return
    args.name = [n for n in args.name if n.strip()]
    try:
        _main(args)
    except SystemExit as e:
        if interactive and e.code not in (None, 0):
            print(e.code)
        elif not interactive:
            raise
    except Exception as e:
        if not interactive:
            raise
        print(f"エラー: {e}")
    if interactive:
        input("\nEnterキーで終了")


def ask(args):
    print("=== キル集 自動モンタージュ ===\n")
    if not args.folder:
        args.folder = input("動画フォルダをこの画面にドラッグして Enter: ").strip().strip('"').strip("'")
    name = input("キルを拾うプレイヤー名 (空欄=音量で自動切り出し): ").strip()
    if name:
        args.name = [n.strip() for n in name.split(",")]
    args.mosaic = input("キルログにモザイクをかける? (y/n): ").strip().lower().startswith("y")
    if input("BGMを使う? フォルダ内の曲を使用 (y/n): ").strip().lower().startswith("n"):
        args.no_music = True
    print()


def _main(args):
    if not args.folder:
        sys.exit("動画フォルダを指定してください")

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ffmpeg / ffprobe が見つかりません。インストールしてPATHを通してください。")
    folder = Path(args.folder.strip().strip('"')).expanduser()
    if not folder.is_dir():
        sys.exit(f"フォルダが見つかりません: {folder}")
    out = Path(args.out) if args.out else folder / "montage.mp4"

    # 以前に作ったモンタージュ (montage*.mp4) は素材にしない
    clips = [p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXT and p.resolve() != out.resolve()
             and not p.stem.lower().startswith("montage")]
    if not clips:
        sys.exit("フォルダに動画がありません")
    if args.order == "date":
        clips.sort(key=lambda p: p.stat().st_mtime)
    elif args.order == "shuffle":
        random.shuffle(clips)
    else:
        clips.sort(key=lambda p: p.name.lower())

    music_files = []
    if not args.no_music:
        music_files = [Path(p) for p in args.music] if args.music else \
            sorted((p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXT), key=lambda p: p.name.lower())
    print(f"動画: {len(clips)} 本")
    print(f"BGM: {', '.join(p.name for p in music_files) if music_files else 'なし'}")
    print(f"モード: {'キルログ (' + ' / '.join(args.name) + ')' if args.name else '音量'}")

    tmp = Path(tempfile.mkdtemp(prefix="montage_"))
    try:
        build(args, folder, out, clips, music_files, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def build(args, folder, out, clips, music_files, tmp):
    fps = args.fps
    W, H = args.res.lower().split("x")

    # ---- 素材区間を決める: [(clip, dur, has_audio, 終了アンカー or None, 自然な長さ)]
    segs = []
    for c in clips:
        dur, has_audio, vw, vh = probe(c)
        if args.name:
            print(f"  {c.name} ({dur / 60:.1f}分) のキルログを解析")
            kills = detect_kills(c, dur, vw, vh, args.name, args,
                                 folder / "_montage_cache", folder / "_feed_debug" if args.debug_feed else None)
            for s, e, n in kills_to_segments(kills, dur, args.pre, args.post, args.merge_gap):
                segs.append(dict(clip=c, dur=dur, audio=has_audio, end=e, length=e - s, kills=n))
            print(f"    → {len(kills)} キル")
        else:
            segs.append(dict(clip=c, dur=dur, audio=has_audio, end=None, length=None, kills=1))
        for s in segs:
            if s["clip"] == c:
                s["vw"], s["vh"] = vw, vh
    if not segs:
        sys.exit("キルが1つも見つかりませんでした。--debug-feed で読み取り範囲を確認してください。")
    if args.detect_only:
        for s in segs:
            print(f"  {s['clip'].name}: {s['end'] - s['length']:.1f}s - {s['end']:.1f}s  ({s['kills']}キル)")
        return

    music = prepare_music(music_files, tmp) if music_files else None
    beats = get_beats(music, args.bpm, args.music_start) if music else None
    music_offset = float(beats[0]) if beats is not None else args.music_start

    # ---- ビートに合わせて長さを確定
    plan, t_frames, bi = [], 0, 0
    for s in segs:
        if beats is not None:
            interval = beats[bi + 1] - beats[bi]
            if s["length"] is None:
                nb = args.beats
            else:
                nb = int(min(args.max_beats, max(2, np.ceil(s["length"] / interval - 0.25))))
            while nb > 1 and beats[bi + nb] - beats[bi] > s["dur"] - 0.05:
                nb -= 1
            end_t = beats[bi + nb] - music_offset
            bi += nb
        else:
            L = s["length"] if s["length"] is not None else min(args.clip_len, s["dur"] - 0.05)
            end_t = t_frames / fps + L
        n_frames = max(1, int(round(end_t * fps)) - t_frames)
        length = n_frames / fps
        if s["end"] is None:
            start = best_window(s["clip"], s["dur"], s["audio"], length, args.peak_pos)
        else:
            # 最後のキルの直後で切れるように、足りない分は前側に伸ばす
            start = min(max(0.0, s["end"] - length), max(0.0, s["dur"] - length))
        plan.append((s["clip"], start, n_frames, s["audio"], s["vw"], s["vh"]))
        t_frames += n_frames
        tag = f" [{s['kills']}キル]" if args.name else ""
        print(f"  {s['clip'].name}: {start:7.2f}s から {length:.2f}s{tag}")
    total = t_frames / fps
    print(f"合計 {total:.1f} 秒 / {len(plan)} カット")

    # ---- 書き出し
    tail = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}")
    regions = ([args.feed_region] if args.mosaic else []) + args.mosaic_region
    parts = []
    for i, (c, start, n_frames, has_audio, vw, vh) in enumerate(plan):
        fc = mosaic_filter(regions, vw, vh, args.mosaic_style, args.mosaic_strength, tail)
        part = tmp / f"part_{i:04d}.mp4"
        length = n_frames / fps
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(c)]
        if not has_audio:
            cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        cmd += ["-map", "0:a:0" if has_audio else "1:a:0",
                "-filter_complex", fc, "-map", "[vout]", "-frames:v", str(n_frames),
                "-af", f"aresample=48000,apad,atrim=0:{length:.4f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(args.crf),
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ac", "2", str(part)]
        print(f"  書き出し {i + 1}/{len(plan)}", end="\r")
        run(cmd)
        parts.append(part)
    print()

    listfile = tmp / "list.txt"
    listfile.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    joined = tmp / "joined.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
         "-c", "copy", str(joined)])

    fade = min(1.5, total / 4)
    fst = max(0.0, total - fade)
    if music:
        af = (f"[0:a]volume={args.game_vol}[g];"
              f"[1:a]aresample=48000,apad,atrim=0:{total:.3f},asetpts=PTS-STARTPTS,volume={args.music_vol}[m];"
              f"[g][m]amix=inputs=2:duration=first:normalize=0,afade=t=out:st={fst:.3f}:d={fade:.3f}[a]")
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(joined), "-ss", f"{music_offset:.3f}", "-i", str(music),
               "-filter_complex", af + f";[0:v]fade=t=out:st={fst:.3f}:d={fade:.3f}[v]",
               "-map", "[v]", "-map", "[a]"]
    else:
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(joined),
               "-vf", f"fade=t=out:st={fst:.3f}:d={fade:.3f}", "-af", f"afade=t=out:st={fst:.3f}:d={fade:.3f}"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
    print("  仕上げ中 (BGM合成・フェードアウト)...")
    run(cmd)
    print(f"\n完成: {out}")


if __name__ == "__main__":
    main()
