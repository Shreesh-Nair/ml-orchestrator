# Ensure shiboken6 runtime pieces are collected
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all('shiboken6')
