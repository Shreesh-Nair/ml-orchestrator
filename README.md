# ML Orchestrator (Binary Classification Demo)

ML Orchestrator is an early-stage no-code ML desktop app focused on a single polished workflow:

- Load a CSV
- Choose a binary target column
- Train and evaluate a RandomForest model
- Save model files and run predictions without writing code

The current app is intentionally scoped to one reliable demo path before expanding features.

## Tech Stack

- Python 3.9-3.12
- PySide6 desktop UI
- scikit-learn
- pandas
- matplotlib

## Quick Start (Source)

1. Install dependencies:

```powershell
poetry install
```

2. Run the GUI:

```powershell
poetry run python -m gui.main
```

3. Optional: run tests:

```powershell
poetry run pytest -q
```

## Demo Workflow

Inside the app:

1. Click `Run Demo Dataset` for a one-click Titanic training run.
2. Review metrics and plots.
3. Click `Save Model`.
4. Go to `Model Library & Predict` and run predictions.

## Data and Artifacts Paths

Runtime resources (read-only app assets):

- `examples/`
- `data/titanic.csv`

User-generated artifacts are stored in:

- `%LOCALAPPDATA%\\ML Orchestrator\\models`
- `%LOCALAPPDATA%\\ML Orchestrator\\logs`
- `%LOCALAPPDATA%\\ML Orchestrator\\pipelines\\generated`

## CLI

List example pipelines:

```powershell
poetry run python -m core.cli list
```

Run an example pipeline:

```powershell
poetry run python -m core.cli run titanic
```

## Build Windows `.exe` (One-Folder)

Prerequisite:

```powershell
poetry add --group dev pyinstaller
```

Build:

```powershell
scripts\\build_demo.ps1 -Clean
```

Expected output:

- `dist\\MLOrchestratorDemo\\MLOrchestratorDemo.exe`

## Build Windows Installer (`.exe` setup)

Prerequisites:

- Built app folder in `dist\\MLOrchestratorDemo`
- Inno Setup 6 installed (`ISCC.exe`)

Build installer:

```powershell
scripts\\build_installer.ps1 -AppVersion 0.1.0
```

Expected output:

- `dist\\installer\\MLOrchestratorDemoSetup.exe`

## Notes

- The GUI runs training in a background worker thread to keep the app responsive.
- Regression/anomaly handlers exist in code, but the GUI demo is intentionally binary-classification-only for now.
