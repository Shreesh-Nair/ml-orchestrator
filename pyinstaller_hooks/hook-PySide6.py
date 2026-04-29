"""Collect PySide6 runtime files and safe-guard optional script modules.

PyInstaller sometimes tries to import optional helper modules such as
``PySide6.scripts.deploy_lib`` during analysis which can raise
``ModuleNotFoundError`` in some PySide6 builds. Explicitly collect package
data and dynamic libs, and collect submodules of the ``scripts`` package if
available while guarding against import errors.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

try:
	# Collect runtime data and dynamic libraries
	datas = collect_data_files("PySide6")
	binaries = collect_dynamic_libs("PySide6")
except Exception:
	datas = []
	binaries = []

hiddenimports = []
try:
	# Collect optional script helpers if present (e.g., deploy_lib)
	from PyInstaller.utils.hooks import collect_submodules

	hiddenimports = collect_submodules("PySide6.scripts")
except Exception:
	# Best-effort: if collect_submodules/import fails, leave hiddenimports empty
	hiddenimports = []
