@echo off
chcp 65001 > nul
rem Python版: 動画フォルダをこのファイルにドラッグ＆ドロップ
python "%~dp0montage.py" %1
