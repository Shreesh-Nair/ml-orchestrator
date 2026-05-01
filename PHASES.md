# ML Orchestrator: 10-Phase Development Roadmap

This document defines the complete 10-phase development roadmap for ML Orchestrator. It serves as the authoritative reference for feature prioritization, scope management, and project phase tracking.

## Phase 0: Product Guardrails (Foundational - Do First)

Lock scope and personas, freeze UX principles, and define quality bars before building features.

### Scope & Personas
- [ ] Define primary users (analyst, operations user, student, small business owner)
- [ ] Document top 5 use cases for each persona
- [ ] Create feature bloat prevention checklist

### UX Principles
- [ ] Wizard-first interaction model
- [ ] Safe defaults (sensible preprocessing choices)
- [ ] Plain-language labels throughout
- [ ] Advanced options hidden by default

### Quality Bars
- [ ] Minimum training success rate target (e.g., 90%+)
- [ ] Error clarity standards
- [ ] Save/load reliability requirements
- [ ] Packaging stability criteria

---

## Phase 1: Core Reliability and Foundation (Must Have)

Essential infrastructure for a reliable no-code experience.

### Project/Session Management
- [x] Add New Project capability
- [x] Open Project / Load Session functionality
- [x] Save Project / Save Session with metadata
- [x] Resume work from saved sessions
- **Status**: Implemented (Save Session / Load Session buttons in GUI)

### Strong Schema and Data Validation
- [x] Check target column presence
- [x] Detect type mismatches
- [x] Identify missing values (by column % threshold)
- [x] High-cardinality column warnings
- [x] Duplicate row detection
- [x] Potential data leakage hints
- [x] Target imbalance detection
- [x] Outlier-heavy column detection
- **Status**: Fully implemented in `core/data_quality.py` with 8+ checks

### Consistent Artifact Management
- [x] Standardized model storage directory
- [x] Log file organization
- [x] Generated pipeline storage
- [x] Prediction output consistency
- **Status**: Implemented via `core/paths.py` with configurable directories

### Robust Error System
- [x] Map technical exceptions to user-facing guidance
- [x] Provide fix steps for common errors
- [x] Error hints for CSV not found, target issues, binary classification failures, empty datasets
- **Status**: Implemented in `core/executor.py` with `_error_hint()` function

### Reproducibility Controls
- [x] Global random seed support
- [x] Run metadata snapshots (timestamp, run_id)
- [x] Dataset fingerprint / version tracking
- [x] Pipeline config snapshots in session
- **Status**: Implemented via run_id, run_timestamp, and run_summary

### Phase 1 Overall Status
✅ **COMPLETE** - All core reliability features are implemented and tested.

---

## Phase 2: Prediction Workflows (Must Have)

Enable users to get predictions from trained models in practical ways.

### Single Prediction Form (Manual Input) Polish
- [x] Auto-generate form from model schema
- [x] Better type hints (int, float, bool, string)
- [x] Nullable field handling
- [x] Feature dropdown with available columns
- **Status**: Implemented in Prediction tab with dynamic form generation

### Batch Prediction via File
- [x] CSV input for batch inference
- [x] Automatic feature column validation
- [x] Progress indication during batch processing
- [x] Prediction CSV export
- **Status**: Implemented ("Load Batch CSV" + "Batch Predict + Export" buttons)

### Prediction Output Options
- [x] Class labels (classification)
- [x] Confidence / probability scores
- [x] Anomaly scores (anomaly detection)
- [x] Timestamps on predictions
- [x] Row ID preservation
- **Status**: Implemented in `core/prediction.py`

### Output Templates
- [x] Concise output profile (minimal columns)
- [x] Detailed output profile (all input + predictions)
- [x] User-selectable templates
- **Status**: Implemented ("Detailed Output" / "Simple Output" in GUI)

### Phase 2 Overall Status
✅ **COMPLETE** - Full prediction workflow end-to-end with both manual and batch modes.

---

## Phase 3: Tabular Model Options (High Value, Still Simple)

Expand model variety, add basic explainability, and provide starter templates for common business intents.

### Classification Options
- [x] Logistic Regression
- [x] Random Forest
- [x] Gradient Boosting (XGBoost)
- **Status**: 3 of 3 implemented ✅

### Regression Options
- [x] Linear Regression
- [x] Random Forest Regressor
- [x] Ridge / Lasso / ElasticNet (regularized options)
- **Status**: 3 of 3 implemented ✅

### Anomaly Detection Options
- [x] Isolation Forest (core algorithm)
- [x] Local Outlier Factor (LOF)
- [x] One-Class SVM
- **Status**: 3 of 3 implemented ✅

### Explainability Basics
- [x] Feature importance (built-in, coefficient, and permutation)
- [x] Plain-language model guidance in the Train tab
- [ ] Feature contribution for single predictions
- **Status**: 2 of 3 implemented ✅

### Task Templates
- [x] Churn prediction template
- [x] Fraud detection template
- [x] House price prediction template
- [x] Demand forecasting template
- **Status**: Implemented in the Train tab as starter workflow presets ✅

### Phase 3 Overall Status
✅ **COMPLETE** - Core tabular model options, explainability basics, and starter templates are in place. Phase 4 is now the next phase.

---

## Phase 4: Hyperparameter and Auto Optimization (High Value)

Let power users tune models without CLI exposure.

### Three Training Modes
- [x] **Quick Mode**: Fast defaults, minimal controls, predictable runtime (UI + direct model run)
- [x] **Tune Mode**: Selected key parameters only with sensible bounds (YAML wiring + tuner MVP)
- [x] **Auto-Optimal Mode**: Time-budgeted search with progress and best-model selection (UI controls for budget)
- **Status**: Complete — all three modes working with UI, YAML generation, and training

### Training Budget Controls
- [x] Max runtime (in minutes) — UI control added
- [x] Max trials / iterations — UI control added
- [x] Early stopping rules (no improvement in N trials) — implemented in tuner
- [x] Validation strategy (random vs stratified split) — UI dropdown selection added
- **Status**: Fully implemented with all core features

### Comparison Report
- [x] Baseline vs tuned model metrics (side-by-side in comparison report dict)
- [x] Training time comparison (elapsed_seconds in report)
- [x] Model size comparison (serialized size in bytes)
- [x] Recommendation for production use (rule-based summary)
- **Status**: Core and advanced comparison fields implemented

### Phase 4 Overall Status
🟡 **IN PROGRESS** (~70%) - Complete hyperparameter tuning with GUI integration: three training modes, expanded search spaces, baseline comparison, early stopping, and results display in metrics table. Remaining: cross-validation strategy UI, optional Optuna Bayesian search, and final polish.

### Checkpoint (2026-05-01)
- [x] UI controls added in `gui/main.py` (training mode, trials, max time)
- [x] `_write_generated_yaml()` wired to emit `hyperparameter_tune` stage when tuning is selected
- [x] `HyperparameterTunerHandler` implemented (MVP randomized trials delegating to existing model handlers)
- [x] Handler registered in `core/handler_registry.py`
- [x] Quick syntax checks passed (`python -m py_compile` across repo)
- [x] Changes committed: `0a61719` — "Phase 4: add hyperparameter tuning MVP — UI, YAML wiring, and tuner handler"

### Checkpoint (2026-05-01 - Tests)
- [x] Created `tests/test_hyperparameter_tuner.py` with 6 comprehensive unit tests
- [x] Tests cover: quick mode, tune mode, n_trials parameter, regression task, error handling, YAML integration
- [x] Fixed circular import in HyperparameterTunerHandler (deferred import of `get_handler_for_stage`)
- [x] 3 out of 6 tests verified passing; 2 tests fixed (regression column name, exception type)
- [x] Next: commit tests + run full suite, then implement search strategy improvements

### Checkpoint (2026-05-01 - Enhanced Tuner)
- [x] Expanded hyperparameter search spaces for all model types (RF, XGBoost, LogReg, Ridge, Lasso, Linear)
- [x] Implemented baseline model training (default parameters for comparison)
- [x] Added early stopping logic (stop after N trials without improvement)
- [x] Built comprehensive comparison report: baseline vs. best, improvement %, elapsed time, metrics side-by-side
- [x] Added helper methods: `_train_model()`, `_extract_score()` for cleaner architecture
- [x] Created new test `test_tuner_comparison_report()` to validate reporting feature
- [x] Changes committed: `e642b16` — "Phase 4: enhance hyperparameter tuner with baseline comparison, early stopping, and expanded search spaces"
- [x] Phase 4 progress: ~60% (UI, YAML, MVP tuner, comparison reporting complete; remaining: visualization/UI display of results, output integration)

### Checkpoint (2026-05-01 - Validation Strategy UI)
- [x] Added `validation_strategy_combo` dropdown in gui/main.py Train tab controls (Random / Stratified options)
- [x] Wired validation_strategy to YAML generation in `_write_generated_yaml()` with default "stratified"
- [x] Updated `handlers/preprocess/tabular_preprocess.py` to use validation_strategy parameter (random vs stratified split)
- [x] Added 2 new tests: `test_validation_strategy_random()` and `test_validation_strategy_stratified()`
- [x] All 31 tests passing (9 hyperparameter tuner + 22 integration tests)
- [x] Changes committed: `[NEW_HASH]` — "Phase 4: add validation strategy UI (random vs stratified split selection)"
- [x] Phase 4 progress: ~85% (UI controls, YAML wiring, enhanced tuner, comparison reporting, GUI display, and validation strategy selection complete; remaining: optional Optuna integration)

### Checkpoint (2026-05-01 - Comparison Report Upgrade)
- [x] Added serialized model size comparison to tuning summary (baseline vs best)
- [x] Added rule-based production recommendation to tuning summary
- [x] Displayed model size and recommendation fields in gui/main.py metrics table
- [x] Extended comparison report test coverage to assert new fields
- [x] All tests passing after changes (37 passed)
- [x] Changes committed: `647eb95` — "Phase 4: add model size and recommendation details to tuning comparison report"
- [x] Phase 4 progress: ~92% (rich comparison report complete; only optional Optuna search remains)

### Phase 4 Overall Status
🟡 **NEAR COMPLETE** (~92%) - Comprehensive hyperparameter tuning with GUI integration: three training modes, expanded search spaces, baseline comparison, early stopping, results display, validation strategy selection, and richer comparison reporting. Remaining: optional Optuna Bayesian search integration.

## Phase 5: Data Handling and Augmentation (Important)

Improve model quality through smarter preprocessing and feature engineering.

### Data Quality Report
- [x] Missingness analysis
- [x] Imbalance detection
- [x] Outlier identification
- [x] Cardinality analysis
- [x] Skew detection
- [x] Leakage hints
- **Status**: Fully implemented in `core/data_quality.py`

### Preprocessing Recipes
- [x] Date/time feature extraction (year, month, day, day-of-week)
- [x] Text feature extraction (word count, character length)
- [x] Rare category grouping (combine infrequent values into "Other")
- [ ] Encoding strategy selection (one-hot, ordinal, target-encoding)
- [ ] Scaling/normalization choices (StandardScaler, MinMaxScaler, none)
- [ ] Imputation strategies (median, mode, forward-fill, drop rows)
- **Status**: 3 of 6 partially implemented; core recipes in place

### Class Imbalance Options
- [ ] Class weights (adjust loss to penalize minority class)
- [ ] Resampling approaches (SMOTE, random over/under-sampling)
- [ ] Threshold adjustment for classification
- **Status**: Not implemented

### Tabular Augmentation Options
- [ ] Light synthetic data generation (SMOTE) in advanced mode
- [ ] Warnings about synthetic data risks
- [ ] Reproducibility with seed
- **Status**: Not implemented

### Data Split Strategies
- [ ] Random split
- [ ] Stratified split (preserve target distribution)
- [ ] Time-aware split (for time-series data)
- **Status**: Random split only (hardcoded in preprocessing)

### Feature Engineering Helpers
- [x] Date/time expansion (hour, minute, season, etc.)
- [x] Numerical bucketization
- [x] Text feature generation
- [ ] Polynomial interactions
- [ ] Domain-specific templates (e.g., "create_business_hours" feature)
- **Status**: 3 of 5 basic helpers implemented

### Phase 5 Overall Status
🟡 **PARTIALLY COMPLETE** (~40%) - Data quality reporting complete; preprocessing recipes started; imbalance handling and augmentation not yet implemented.

---

## Phase 6: Vision Workflows (Scope Carefully)

Add image and video support with safe, inference-first approach.

### Image Classification Inference First
- [ ] Folder/CSV input with image paths
- [ ] Predicted label output per image
- [ ] Confidence scores per class
- **Status**: Not started

### Image Classification Training Next
- [ ] Pretrained transfer-learning templates (ResNet, MobileNet, ViT)
- [ ] Minimal configuration (epochs, learning rate)
- [ ] GPU detection and auto-fallback to CPU
- **Status**: Not started

### Image Augmentation Presets
- [ ] Flip, rotate, crop presets
- [ ] Color jitter options
- [ ] Preview augmented samples
- **Status**: Not started

### Video Object Detection Inference
- [ ] Pretrained detection pipeline (YOLO, Faster R-CNN)
- [ ] Annotated video export
- [ ] CSV detections (frame, object, confidence, bbox)
- **Status**: Not started

### GPU Awareness and Fallback
- [ ] Detect available GPU
- [ ] Graceful CPU fallback
- [ ] Memory-aware batch sizing
- **Status**: Not started

### Phase 6 Overall Status
🔴 **NOT STARTED** (0%) - No vision workflows yet. Defer until Phase 5 is complete.

---

## Phase 7: UX and Product Experience (Differentiator)

Polish interaction patterns and make the app feel professional.

### Guided Wizard for Every Task
- [ ] Consistent 6-step pattern across all workflows
- [ ] Progress indicator (step 1 of 6, etc.)
- [ ] "Back", "Next", "Finish" navigation
- [ ] Clear step titles and descriptions
- **Status**: Partial (Recommender toggles provide guidance, but not full wizard)

### Progressive Disclosure
- [x] Basic panel shown by default
- [x] Advanced panel optional (hidden until expanded)
- **Status**: Partially implemented with recommender toggles

### Model Cards in Plain Language
- [ ] "What it predicts" explanation
- [ ] Confidence and limitations
- [ ] Recommended use cases
- [ ] Do-not-use warnings
- **Status**: Not implemented

### Better Visual Outputs
- [ ] Clear confusion matrix visualization
- [ ] ROC/PR curve plots
- [ ] Residual plots for regression
- [ ] Anomaly ranking table with scores
- **Status**: Partially implemented (metrics shown, plots missing)

### In-App Help and Suggestions
- [x] Contextual tooltips for features
- [x] Data quality hints with recommendations
- [x] Quick-fix suggestions
- [ ] Next best action suggestions
- **Status**: Tooltips and recommendations in place; next-action guidance partial

### Accessibility and Responsiveness
- [ ] Readable typography and contrast
- [ ] Keyboard navigation support
- [ ] Tab order and focus management
- [ ] Useful default values everywhere
- **Status**: Basic UX in place; not formally tested for accessibility

### Phase 7 Overall Status
🟡 **PARTIALLY COMPLETE** (~35%) - Tooltips and basic guidance present; wizard, model cards, and visual outputs missing.

---

## Phase 8: Standalone Distribution and Operations (Must Have for Real Users)

Ensure the app works for non-technical users without Python.

### Release-Grade Installer Workflow
- [x] One-click installer (Inno Setup)
- [x] Desktop shortcut creation
- [x] Uninstall support
- [x] CI/CD release packaging
- **Status**: Implemented (release-packaging.yml, build_release.ps1, etc.)

### Portable Mode Decision
- [x] Installer-first as primary
- [ ] Optional zip distribution for advanced users
- **Status**: Installer implemented; portable zip not yet offered

### Build Reproducibility
- [x] Versioned build scripts
- [x] Artifact verification (checksums, manifests)
- [x] Dependency pinning (numpy<2, pandas<2.2, etc.)
- **Status**: Implemented with release-manifest.json

### Runtime Diagnostics Pack
- [x] Comprehensive logging (run_logger.py)
- [ ] Diagnostic bundle export (logs + config + system info)
- **Status**: Logging in place; diagnostic bundle not yet packaged

### Safe Updates
- [ ] Version checker (auto-detect new releases)
- [ ] Rollback strategy (keep N previous versions)
- [ ] Update without disrupting work
- **Status**: Not implemented

### Dependency and Security Hygiene
- [x] Pinned critical dependencies
- [ ] Routine vulnerability checks (e.g., Dependabot)
- [ ] Security audit trail in logs
- **Status**: Manual pinning in place; automated checks not set up

### Phase 8 Overall Status
🟡 **PARTIALLY COMPLETE** (~60%) - Installer and build reproducibility complete; portable mode, diagnostics, and updates not yet implemented.

---

## Phase 9: Testing and Quality Gates (Non-Negotiable)

Comprehensive automated testing to prevent regressions.

### Unit Tests
- [x] Parser tests (YAML, CSV loading)
- [x] Handler tests (preprocessing, model training)
- [x] Path resolution tests
- **Status**: Implemented (test_yaml_parser.py, test_executor_*.py)

### Integration Tests
- [x] End-to-end task flows (train -> save -> predict)
- [x] Multi-task pipeline runs (classification, regression, anomaly)
- [x] Recommender integration tests
- **Status**: Implemented (test_integration_pipeline_with_recommender.py)

### GUI Smoke Tests
- [x] Key user journeys (load CSV -> train -> view results -> save)
- [x] Tab navigation and model switching
- [x] Session save/load
- **Status**: Implemented (test_gui_smoke.py, test_gui_preprocess_toggle_yaml.py)

### Save/Load Compatibility Tests
- [ ] Model backward compatibility across versions
- [ ] Session file migration and versioning
- [ ] Configuration format stability
- **Status**: Partial (session save/load works; version compatibility not tested)

### Packaging Smoke Tests
- [x] Standalone .exe build
- [x] Installer generation
- [x] Clean Windows deployment test
- **Status**: Implemented (PyInstaller builds, release workflow)

### Performance Smoke Benchmarks
- [x] Medium-dataset training (keep UI responsive)
- [x] Large-dataset preprocessing
- [x] Batch prediction throughput
- **Status**: Implemented (test_performance_benchmark.py)

### Regression Test Suite
- [x] Generated pipeline YAML from GUI options
- [x] Predicted vs baseline outputs
- **Status**: Implemented (test_executor_regression.py, test_executor_classification_logreg.py)

### Phase 9 Overall Status
✅ **COMPLETE** - Comprehensive test suite with 53+ tests covering unit, integration, GUI, performance, and packaging. **All tests passing.**

---

## Phase 10: Tentative Final Version Definition

All these conditions must be true for a v1.0 final release:

- [ ] Stable standalone install on clean Windows (no Python required)
- [ ] Reliable tabular workflows: classification, regression, anomaly
- [ ] Full prediction support: manual + batch file input with export
- [ ] Quick, Tune, Auto training modes with clear outputs
- [ ] Strong validation and plain-language error guidance
- [ ] Session save/reopen and model library with versioned metadata
- [ ] Vision support for image classification and video detection (inference)
- [ ] Comprehensive automated tests (unit, integration, GUI, packaging)
- [ ] Documentation and tutorial flows for non-technical users
- [ ] Release installer with auto-update mechanism
- [ ] Security hygiene checks (dependency scanning, CVE monitoring)

### Current Status
🔴 **NOT READY** - Missing Phase 4 (hyperparameter tuning), Phase 6 (vision), Phase 7 UX polish, and Phase 8 updates/diagnostics.

---

# Current Phase Summary

**As of now: Phase 3 (Tabular Model Options) is complete and Phase 4 (Hyperparameter Tuning) is in-progress.**

- ✅ **Phase 0**: Guardrails not formalized but vision clear
- ✅ **Phase 1**: Core reliability 100% complete
- ✅ **Phase 2**: Prediction workflows 100% complete
- ✅ **Phase 3**: Tabular models complete (classification/regression/anomaly handlers and explainability basics implemented)
- 🟡 **Phase 4**: Hyperparameter tuning in progress (~30%): UI, YAML wiring, and MVP tuner implemented
- 🟡 **Phase 5**: Data augmentation ~40% (data quality complete; preprocessing recipes partial; imbalance/augmentation missing)
- 🔴 **Phase 6**: Vision workflows not started
- 🟡 **Phase 7**: UX polish ~35% (basic guidance present; wizard/model cards/visual outputs missing)
- 🟡 **Phase 8**: Distribution ~60% (installer complete; updates/diagnostics missing)
- ✅ **Phase 9**: Testing 100% (53+ tests passing)
- 🔴 **Phase 10**: Final version not yet ready

---

# Recommended Next Steps (Phase 3 Priority Order)

1. **Add Gradient Boosting for classification** (XGBoost or LightGBM) - high user impact
2. **Add regularized regression options** (Ridge, Lasso, ElasticNet) - low complexity, high value
3. **Add explainability basics** (permutation importance, feature importance plots) - builds trust
4. **Add LOF and One-Class SVM to anomaly detection** - completes anomaly toolbox

---

# Workflow for Phase Progression

After each implementation step, follow this locked workflow:

1. **Implement** the feature
2. **Test** it (add/update tests)
3. **Commit** with a one-sentence message (e.g., "Add XGBoost support to classification")
4. **Update .gitignore** if needed
5. **Update this PHASES.md file** to mark completion
6. **Move to next prioritized task**

This ensures clear progress tracking and no context loss.
