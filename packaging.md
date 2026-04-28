# Reproducible Packaging Guide

This document describes a reproducible packaging environment and a CI precheck to avoid NumPy ABI incompatibilities when building a Windows executable with PyInstaller.

Why: Many binary packages (SciPy, scikit-learn, pandas) include compiled extensions built against a particular NumPy ABI. Building a bundled executable in an environment with a different major NumPy version (e.g. NumPy 2.x vs 1.x) can cause runtime errors and PyInstaller warnings referencing `_multiarray_umath`.

Recommended approach (short):

1. Create a dedicated packaging environment (Conda recommended on Windows):

```powershell
conda create -n ml-orch-pack python=3.12 -y
conda activate ml-orch-pack
conda install pip -y
```

2. Install pinned packages for packaging (this matches our CI):

```powershell
pip install "numpy<2" "pandas<2.2" "scikit-learn==1.5.2" pyinstaller "PySide6==6.6.*"
# then install other runtime deps as needed (or use requirements file)
pip install -r requirements.txt
```

3. Verify compiled extensions and NumPy version quickly (script included):

```powershell
python scripts\find_compiled_extensions.py
# or just print NumPy version
python -c "import numpy as np; print(np.__version__)"
```

4. Build executable (Windows):

```powershell
pyinstaller --noconfirm --clean ml_orchestrator_demo.spec
```

Notes and longer-term options:

- Short-term reliable option: pin `numpy<2` in the packaging environment (what our CI does). This avoids ABI mismatch warnings for prebuilt wheels.
- Long-term: upgrade all compiled wheels (scipy, scikit-learn, pandas) to NumPy 2-compatible releases or rebuild those wheels from source against NumPy 2. This requires platform-specific wheel availability or an internal build process.
- Docker: building a Windows exe inside Docker requires Windows containers — CI uses `windows-latest` runners. For cross-platform packaging of non-Windows artifacts (Linux/macOS), you can create Docker images pinned to the same package versions.

Troubleshooting:

- If PyInstaller logs show warnings like `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`, inspect the output of `scripts/find_compiled_extensions.py` to find which packages include .pyd/.dll files and confirm their compatibility.
- If you must build using NumPy 2, ensure all compiled dependencies are available as NumPy 2-compatible wheels for your Python version, or rebuild them.

Additional helper files:
- `scripts/find_compiled_extensions.py` — scans installed site-packages and writes `build/compiled_extensions.json`.
- `.github/workflows/packaging_check.yml` — simple CI job that checks the packaging environment's NumPy major version (fails if >= 2) to prevent accidental packaging with NumPy 2.

If you want, I can also add a `packaging/conda-env.yml` and a `Dockerfile` to produce identical environments across machines.
