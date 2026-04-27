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

## Data Quality Report

The `Train` tab now includes a built-in data quality panel.

1. Load any CSV dataset.
2. Select a target column.
3. Review summary and warnings for:
	- Missing values
	- Duplicate rows
	- Constant columns
	- High-cardinality columns
	- Potential ID-like columns
	- Target imbalance
	- Outlier-heavy numeric columns
4. (Optional) Apply `Quick Fixes`:
	- Drop duplicate rows
	- Drop constant columns
	- Handle missing values with `None`, `Drop rows`, or `Fill simple (median/mode)`
	- Use `Preview Quick Fixes` to inspect impact without changing data
	- Use `Apply Quick Fixes` and confirm to execute changes
	- Use `Revert Quick Fixes` to return to the original loaded dataset
5. Click `Export Quality Report` to save a JSON report.

By default, the save dialog points to your local exports directory under `%LOCALAPPDATA%\ML Orchestrator\exports`.
Quick-fix selections are saved in session files and restored on load.

## Batch Predict + Export

You can run file-based prediction workflows for classification, regression, and anomaly models.

1. Train and save a model from the `Train` tab.
2. Open `Model Library & Predict` and select the saved model.
3. Click `Load Batch CSV` and choose an input file containing the required feature columns.
4. Choose export profile:
	- `Detailed Output`: keeps input columns and adds prediction outputs.
	- `Simple Output`: exports only essential prediction columns.
5. Click `Batch Predict + Export` and save the output CSV.

Output behavior by task:

1. Classification:
	- Detailed: input columns + `prediction` + probability columns.
	- Simple: `prediction` + `prediction_score`.
2. Regression:
	- Detailed: input columns + `prediction`.
	- Simple: `prediction`.
3. Anomaly:
	- Detailed: input columns + `prediction` + `anomaly_score`.
	- Simple: `prediction` + `anomaly_score`.

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

## Product Direction (Simple but Robust)

This project is a strong idea, especially now. Many AI tools generate demos quickly, but very few become reliable desktop products that non-technical users can trust day to day.

To stay practical and avoid complexity creep:

1. Keep a wizard-first UX for all common workflows.
2. Expose complexity progressively (Basic mode by default, Advanced optional).
3. Prioritize reliability, validation, and clear error guidance over adding many toggles.
4. Add new ML options as templates, not as one giant settings page.
5. Build once, run anywhere on Windows without requiring Python installs.

## Recommended Feature Roadmap

The goal is to add many ML options while keeping the product easy to use.

### Phase 1 - Core Robustness (Highest Priority)

1. Project/session files so users can reopen and continue work.
2. Better input validation and plain-language fix suggestions (missing columns, wrong target type, class imbalance).
3. Predict tab improvements:
	- Manual single-row input (already present, improve UX).
	- Batch prediction via CSV.
	- Output file export (CSV with predictions and probabilities/scores).
4. Model lifecycle basics:
	- Save/load model metadata and schema checks.
	- Version labels for saved models.
5. Reproducibility controls:
	- Global random seed option.
	- Run summary with config + metrics + artifact paths.

### Phase 2 - Tabular ML Expansion

1. Classification templates:
	- Logistic Regression
	- Random Forest
	- Gradient Boosting / XGBoost-style option (if dependency choice permits)
2. Regression templates:
	- Linear Regression
	- Random Forest Regressor
	- ElasticNet/Ridge/Lasso options
3. Anomaly detection templates:
	- Isolation Forest
	- One-Class SVM (optional advanced)
	- Local Outlier Factor (optional advanced)
4. Hyperparameter modes:
	- Quick (safe defaults)
	- Tune (grid/random search with time budget)
	- Auto (best effort optimizer for selected task)

### Phase 3 - Data Handling and Augmentation

1. Built-in data quality report:
	- Missing values, cardinality, outliers, leakage warnings.
2. Feature preparation helpers:
	- Date/time extraction
	- Text basics (length, TF-IDF template)
	- Categorical grouping for rare classes
3. Augmentation options by modality:
	- Tabular: noise/jitter and class balancing (SMOTE-like options where appropriate)
	- Image: flip/rotate/crop/brightness presets
	- Time-series: windowing and scaling templates
4. Data split policies:
	- Random split
	- Stratified split
	- Time-aware split

### Phase 4 - Computer Vision Workflows

1. Image classification workflow (folder-to-label and CSV label formats).
2. Image augmentation pipeline with preview.
3. Batch image inference with output CSV + copied/annotated outputs.
4. Video object detection workflow:
	- Pretrained model inference mode first.
	- Export annotated video and detection CSV.
5. Hardware awareness:
	- Auto-detect CPU/GPU capability.
	- Graceful fallback and performance warnings.

### Phase 5 - Usability and Trust Features

1. Template gallery with plain-language use cases:
	- Churn prediction
	- Fraud detection
	- House price prediction
	- Defect image classification
2. Model cards in plain language:
	- What the model predicts
	- Confidence and limitations
	- Recommended usage boundaries
3. Explainability basics:
	- Feature importance
	- Per-row explanation summary (lightweight)
4. Safer release channel:
	- In-app version check
	- Rollback-friendly installer strategy

## Suggested UX Pattern (Keep It Easy)

For every task, keep the same 6-step flow:

1. Choose task template.
2. Load data (manual file or folder).
3. Validate and auto-fix suggestions.
4. Train (Quick/Tune/Auto).
5. Review metrics and artifacts.
6. Predict (manual input or batch file), then export results.

This consistency lets you support many ML options without overwhelming users.

## Why This Project Is Worth Building

Yes, this is a good project in the current AI landscape.

Reasons it stands out:

1. It targets real users who need outcomes, not code.
2. It runs locally, which helps with privacy and reliability.
3. It can become a practical "ML productivity tool" instead of a one-off demo.
4. The existing architecture (pipeline stages + handlers + GUI) is a solid foundation for modular growth.

## Master Backlog To Tentative Final Version

This section is the full feature/change checklist from the current demo to a tentative final version.

### Phase 0 - Product Guardrails

Phase 0 is now locked to prevent feature bloat and keep implementation decisions consistent.

#### 0.1 Primary Users (Locked)

1. Analyst
	- Goal: run fast tabular experiments and export predictions for reporting.
	- Constraint: wants speed and reproducibility, does not want to code.
2. Operations User
	- Goal: run repeatable prediction jobs on new files with minimal setup.
	- Constraint: needs stable outputs and clear failure messages.
3. Student / Learner
	- Goal: understand ML flow through templates and visual feedback.
	- Constraint: easily overwhelmed by excessive options.
4. Small Business Owner
	- Goal: get practical predictions for business decisions without hiring an ML team.
	- Constraint: limited time, expects one-click defaults that work.

#### 0.2 Top 5 Use Cases (Locked)

1. Binary classification from CSV with model save and reload.
2. Regression prediction from CSV with exported result file.
3. Anomaly detection from CSV with scored output file.
4. Manual single-record prediction form for quick what-if checks.
5. Batch prediction on a user-provided CSV, exporting a new CSV with predictions and confidence/scores.

All roadmap items should map to at least one of these 5 use cases or be deferred.

#### 0.3 UX Principles (Frozen)

1. Wizard-first
	- Every workflow follows a guided sequence with visible progress.
2. Safe defaults
	- The default path should produce usable results without advanced tuning.
3. Plain-language labels
	- Prefer user language ("Target column") over technical jargon.
4. Progressive disclosure
	- Advanced controls are hidden by default and shown only when requested.
5. Consistent interaction model
	- Same flow pattern across classification, regression, anomaly, and future vision tasks.
6. Actionable errors
	- Every error should include what failed and what the user should do next.

#### 0.4 Quality Bars (Minimum Acceptance Standards)

1. Training success rate
	- On bundled templates and known-good datasets, successful end-to-end runs must be >= 95% across test executions.
2. Error clarity
	- For common failures (missing columns, wrong target type, unreadable file, invalid model file), user-facing errors must explain cause and next step in one message.
3. Save/load reliability
	- Model save and reload must preserve schema compatibility and allow successful prediction on valid inputs in >= 99% of automated compatibility tests.
4. Packaging stability
	- Installer and one-folder build must launch on a clean Windows machine with no Python preinstalled.
5. Output integrity
	- Exported prediction files must retain input row order and append predictions/confidence without silent row drops.
6. Responsiveness baseline
	- UI must remain responsive during training/inference via background execution for supported workflows.

#### 0.5 Scope Filter (Decision Rule)

A new feature is accepted only if it satisfies all checks below:

1. Helps at least one locked persona complete one locked use case.
2. Fits wizard-first UX without forcing advanced complexity into the default path.
3. Does not reduce Phase 0 quality bars.
4. Can be tested with deterministic acceptance criteria.

### Phase 1 - Core Reliability Foundation

1. Add project/session management (new, save, open, resume).
2. Add stronger data validation and schema checks (target type, missing columns, leakage hints, class imbalance warnings).
3. Standardize artifact handling (models, logs, generated pipelines, exports).
4. Improve user-facing error messages with clear next steps.
5. Add reproducibility controls (seed, config snapshot, run metadata summary).

### Phase 2 - Prediction Workflows

1. Improve manual single-row prediction UX.
2. Add batch prediction from CSV.
3. Export prediction outputs to CSV (labels, confidence/probability, anomaly scores, row ids).
4. Add output profile presets (simple vs detailed output columns).

### Phase 3 - Tabular ML Expansion

1. Classification options: Logistic Regression, Random Forest, Gradient Boosting.
2. Regression options: Linear Regression, Random Forest Regressor, Ridge/Lasso/ElasticNet.
3. Anomaly options: Isolation Forest, LOF, One-Class SVM (advanced).
4. Explainability baseline: feature importance and lightweight per-row explanations.
5. Template gallery entries: churn, fraud, house price, customer risk, demand forecasting.

### Phase 4 - Hyperparameter and Auto Optimization

1. Add three training modes: Quick, Tune, Auto-Optimal.
2. Quick mode uses stable defaults and low runtime.
3. Tune mode exposes selected parameters with safe ranges and time budget.
4. Auto-Optimal mode runs best-effort search and selects best model.
5. Add training budget controls (max time, max trials, early stop).
6. Show side-by-side baseline vs tuned metrics.

### Phase 5 - Data Handling and Augmentation

1. Built-in data quality report (missingness, outliers, cardinality, skew, leakage hints).
2. Feature helpers (date expansion, text basics, rare-category handling).
3. Class imbalance strategies (weights, resampling options).
4. Tabular augmentation options with safety guidance.
5. Split policies: random, stratified, time-aware.

### Phase 6 - Vision Workflows

1. Image classification inference workflow first.
2. Image classification training workflow using transfer-learning templates.
3. Image augmentation presets with preview.
4. Video object detection inference workflow.
5. Export outputs for vision workflows (annotated images/videos and CSV detections).
6. Hardware awareness and fallback (GPU detect, CPU fallback, runtime warnings).

### Phase 7 - Usability and Trust Enhancements

1. Keep a consistent 6-step wizard flow across tasks.
2. Add model cards in plain language (what, confidence, limitations).
3. Improve visual reporting (ROC/PR, residuals, confusion matrix, anomaly ranking).
4. Add in-app help and contextual next-step suggestions.
5. Add accessibility improvements (readability, keyboard support, simple defaults).

### Phase 8 - Standalone Distribution and Operations

1. Keep installer-first distribution for non-technical users.
2. Add optional portable zip distribution for advanced users.
3. Add build reproducibility checks and release validation scripts.
4. Add diagnostics bundle export for support cases.
5. Add safe update strategy and rollback guidance.
6. Maintain dependency pinning and vulnerability checks.

### Phase 9 - Testing and Quality Gates

1. Expand unit tests for handlers, parser, and paths.
2. Add full integration tests for major workflows.
3. Add GUI smoke tests for key user journeys.
4. Add model save/load compatibility tests across versions.
5. Add packaging smoke tests on clean Windows machines.
6. Add performance checks for medium datasets.
7. Add regression tests for generated pipeline configs.

### Phase 10 - Tentative Final Version Definition

A tentative final version is reached when all of the following are true:

1. Clean Windows install works without Python or manual dependency setup.
2. Tabular workflows are stable for classification, regression, and anomaly tasks.
3. Prediction supports both manual input and batch file input with export.
4. Quick, Tune, and Auto-Optimal training modes are stable and understandable.
5. Validation and error guidance are strong enough for non-technical users.
6. Project/session management and model library are robust.
7. Vision workflows support at least image classification and video detection inference.
8. Automated tests and packaging checks are release-grade.
9. Documentation and in-app guidance are sufficient for first-time users.

### Recommended Implementation Order

1. Phase 1 + Phase 2
2. Phase 3 + Phase 4
3. Phase 8 + Phase 9
4. Phase 5
5. Phase 6
6. Phase 7
7. Final hardening against Phase 10 checklist

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

- Shreesh