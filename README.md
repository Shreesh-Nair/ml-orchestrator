# ML Orchestrator (No-Code Desktop ML App)

ML Orchestrator is a no-code desktop ML app for people who are not into coding or machine learning.

The long-term goal is a standalone app experience like VLC or WinRAR: install once, open data, run ML workflows, and get usable outputs with clear guidance.

## Product Vision

- No-code first: users should complete common ML tasks without writing code.
- Standalone desktop UX: simple install, predictable behavior, and local-first workflow.
- Safe defaults: sensible preprocessing and modeling choices out of the box.
- Guided learning: plain-language errors, explanations, and next steps inside the app.

## Current Scope (v0.1.0 Demo)

The current app is intentionally scoped to one polished flow before expanding features:

- Load a CSV
- Choose a binary target column
- Train and evaluate a RandomForest model
- Save model files and run predictions

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

3. Run tests:

```powershell
poetry run pytest -q
```

If Poetry is not installed, run tests directly:

```powershell
pytest -q
```

## Demo Workflow

Inside the app:

1. Click `Run Demo Dataset` for a one-click Titanic training run.
2. Review metrics and plots.
3. Click `Save Model`.
4. Go to `Model Library & Predict` and run predictions.

## Testing Status

Current automated tests cover:

- YAML parsing and validation (`tests/test_yaml_parser.py`)
- Pipeline execution for Titanic and fraud examples (`tests/test_executor_titanic.py`, `tests/test_executor_fraud.py`)
- End-to-end binary MVP pipeline run (`tests/test_binary_mvp_flow.py`)
- Run logging behavior and CSV path resolution (`tests/test_logging_and_path_resolution.py`)

This is a strong base for core logic, but it is not yet sufficient for a production-grade standalone desktop app.

## Recommended Test Expansion

Priority 1:

1. GUI smoke tests for core user journey (load CSV -> train -> view metrics -> save model -> load model -> predict).
2. Model save/load compatibility tests (including backward compatibility for older saved model files).
3. Failure-path tests (bad CSV schema, missing values, wrong target type, file permission errors).

Priority 2:

1. CLI behavior tests (`list`, `run`, invalid arguments, missing pipelines).
2. Resource/path tests for packaged mode (`sys.frozen`, `_MEIPASS`, local app data paths).
3. Regression tests for generated pipeline YAML from GUI options.

Priority 3:

1. Packaging smoke tests for built `.exe` and installer output structure.
2. Performance tests on medium-size datasets to keep UI responsive.
3. Basic reproducibility checks (fixed random state where appropriate).

## Recommended Product Changes (Keeping UX Simple)

1. Add a template gallery with plain language entry points: "Churn", "Fraud", "House Price", "Customer Segments".
2. Add a step-by-step wizard mode so first-time users do not see too many options at once.
3. Add beginner-safe defaults and an "Advanced" panel for optional tuning.
4. Add strong data checks with actionable messages before training starts.
5. Add model cards after training with plain-language interpretation of metrics and risks.
6. Add a stable project/session concept so users can reopen work without confusion.
7. Add an update and rollback strategy for desktop releases to reduce support issues.

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
- Regression and anomaly handlers exist in code, but the GUI demo is intentionally binary-classification-only for now.
