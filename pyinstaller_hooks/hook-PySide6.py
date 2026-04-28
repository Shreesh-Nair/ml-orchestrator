# Custom hook to ensure PySide6 data files, binaries and hiddenimports are collected
from PyInstaller.utils.hooks import collect_all

# collect_all returns (datas, binaries, hiddenimports)
datas, binaries, hiddenimports = collect_all('PySide6')

# Expose to PyInstaller
# 'datas' and 'binaries' will be added automatically; hiddenimports appended
