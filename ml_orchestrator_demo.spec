# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path


project_root = Path(SPECPATH).resolve()

datas = [
    (str(project_root / "examples"), "examples"),
    (str(project_root / "data" / "titanic.csv"), "data"),
    (str(project_root / "LICENSE"), "."),
]

hiddenimports = [
    "matplotlib.backends.backend_qtagg",
    # Common runtime modules that PyInstaller sometimes misses for Qt and NumPy
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "shiboken6",
    "numpy.core._multiarray_umath",
]


a = Analysis(
    ["gui/main.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    # Project-specific hooks to ensure PySide6/shiboken and related Qt plugins are collected
    hookspath=[str(project_root / "pyinstaller_hooks")],
    hooksconfig={"matplotlib": {"backends": ["QtAgg"]}},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "tkinter",
        "torch",
        "torchaudio",
        "torchvision",
        "matplotlib.tests",
        "numpy.tests",
        "pandas.tests",
        "scipy.tests",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MLOrchestratorDemo",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="MLOrchestratorDemo",
)
