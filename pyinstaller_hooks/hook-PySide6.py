"""Collect PySide6 runtime files without forcing a full package import.

The standard PySide6 hooks handle the Qt submodules imported by the app.
This hook only gathers package data and binaries so PyInstaller does not
need to import optional PySide6 script helpers during analysis.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = collect_data_files("PySide6")
binaries = collect_dynamic_libs("PySide6")
hiddenimports = []
