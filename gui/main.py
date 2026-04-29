from __future__ import annotations

import datetime
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.executor import run_pipeline
from core.data_quality import analyze_data_quality, apply_quick_fixes, write_data_quality_report, recommend_quick_fixes
from core.prediction import predict_dataframe
from core.paths import (
    get_data_dir,
    get_demo_dataset_path,
    get_exports_dir,
    get_generated_pipelines_dir,
    get_models_dir,
    get_projects_dir,
)

matplotlib.use("QtAgg")

DATA_DIR = get_data_dir()
GENERATED_DIR = get_generated_pipelines_dir()
MODELS_DIR = get_models_dir()
PROJECTS_DIR = get_projects_dir()
EXPORTS_DIR = get_exports_dir()


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width: float = 5.0, height: float = 4.0, dpi: int = 100):
        figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = figure.add_subplot(111)
        super().__init__(figure)


class PipelineRunWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, yaml_path: str) -> None:
        super().__init__()
        self.yaml_path = yaml_path

    def run(self) -> None:
        try:
            context = run_pipeline(self.yaml_path)
            self.succeeded.emit(context)
        except Exception as exc:
            self.failed.emit(str(exc))


TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "classification": {
        "label": "Classification",
        "models": [
            ("Random Forest", "classification_rf"),
            ("Logistic Regression", "classification_logreg"),
        ],
        "require_binary_target": True,
    },
    "regression": {
        "label": "Regression",
        "models": [
            ("Random Forest Regressor", "regression_rf"),
            ("Linear Regression", "regression_linear"),
        ],
        "require_binary_target": False,
    },
    "anomaly": {
        "label": "Anomaly Detection",
        "models": [
            ("Isolation Forest", "anomaly_isolation_forest"),
        ],
        "require_binary_target": True,
    },
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ML Orchestrator - Binary Classification Demo")

        self.csv_path: Path | None = None
        self.training_csv_path: Path | None = None
        self.original_df: pd.DataFrame | None = None
        self.target_column: str | None = None
        self.current_df: pd.DataFrame | None = None

        self.last_run_context: Dict[str, Any] | None = None
        self.loaded_model_payload: Dict[str, Any] | None = None
        self.current_session_path: Path | None = None

        self.prediction_inputs: Dict[str, Any] = {}
        self.prediction_schema: List[Dict[str, str]] = []
        self.prediction_allowed_values: Dict[str, List[str]] = {}
        self.batch_input_df: pd.DataFrame | None = None
        self.batch_input_path: Path | None = None
        self.run_worker: PipelineRunWorker | None = None
        self.data_quality_report: Dict[str, Any] | None = None
        self._auto_recommendations: Dict[str, Any] | None = None
        self._auto_preprocess_recommendations: Dict[str, Any] = {}

        self.selected_task: str = "classification"
        self.selected_model_type: str = "classification_rf"

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        train_tab = QWidget()
        self.setup_train_tab(train_tab)
        self.tabs.addTab(train_tab, "Train")

        predict_tab = QWidget()
        self.setup_predict_tab(predict_tab)
        self.tabs.addTab(predict_tab, "Model Library & Predict")

        # Preferences file path for storing UI choices
        from pathlib import Path as _Path

        self._prefs_dir = _Path.home() / ".ml-orchestrator"
        self._prefs_path = self._prefs_dir / "prefs.json"

        # Load persisted preferences (if any)
        try:
            self._load_preferences()
        except Exception:
            # fail silently; prefs are optional
            pass

    def setup_train_tab(self, tab: QWidget) -> None:
        splitter = QSplitter(Qt.Vertical)
        layout = QVBoxLayout(tab)
        layout.addWidget(splitter)

        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)

        file_row = QHBoxLayout()
        self.file_label = QLabel("No CSV selected")
        self.btn_browse_csv = QPushButton("Browse CSV")
        self.btn_browse_csv.clicked.connect(self.on_browse_clicked)
        self.btn_run_demo = QPushButton("Run Demo Dataset")
        self.btn_run_demo.clicked.connect(self.on_run_demo_clicked)
        file_row.addWidget(self.file_label)
        file_row.addWidget(self.btn_browse_csv)
        file_row.addWidget(self.btn_run_demo)
        top_layout.addLayout(file_row)

        self.lbl_training_source = QLabel("Training source: -")
        self.lbl_training_source.setStyleSheet("font-size: 12px; color: #555;")
        self.lbl_training_source.setWordWrap(True)
        top_layout.addWidget(self.lbl_training_source)

        session_row = QHBoxLayout()
        self.btn_save_session = QPushButton("Save Session")
        self.btn_save_session.clicked.connect(self.on_save_session_clicked)
        self.btn_load_session = QPushButton("Load Session")
        self.btn_load_session.clicked.connect(self.on_load_session_clicked)
        session_row.addWidget(self.btn_save_session)
        session_row.addWidget(self.btn_load_session)
        session_row.addStretch()
        top_layout.addLayout(session_row)

        task_row = QHBoxLayout()
        task_row.addWidget(QLabel("Task:"))
        self.task_combo = QComboBox()
        self.task_combo.addItem("Classification", "classification")
        self.task_combo.addItem("Regression", "regression")
        self.task_combo.addItem("Anomaly Detection", "anomaly")
        self.task_combo.currentIndexChanged.connect(self.on_task_changed)
        task_row.addWidget(self.task_combo)
        task_row.addStretch()
        top_layout.addLayout(task_row)

        algo_row = QHBoxLayout()
        algo_row.addWidget(QLabel("Algorithm:"))
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        algo_row.addWidget(self.model_combo)
        algo_row.addStretch()
        top_layout.addLayout(algo_row)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target column:"))
        self.target_combo = QComboBox()
        self.target_combo.setEnabled(False)
        self.target_combo.currentIndexChanged.connect(self.on_target_changed)
        target_row.addWidget(self.target_combo)
        top_layout.addLayout(target_row)

        quality_group = QGroupBox("Data Quality")
        quality_layout = QVBoxLayout()
        self.lbl_data_quality_summary = QLabel("Load a dataset to see quality summary.")
        self.lbl_data_quality_summary.setWordWrap(True)
        self.lbl_data_quality_summary.setStyleSheet("font-size: 12px; color: #333;")
        quality_layout.addWidget(self.lbl_data_quality_summary)

        self.lbl_data_quality_warnings = QLabel("")
        self.lbl_data_quality_warnings.setWordWrap(True)
        self.lbl_data_quality_warnings.setStyleSheet("font-size: 12px; color: #a65f00;")
        quality_layout.addWidget(self.lbl_data_quality_warnings)

        # Show auto-detected preprocess recommendations
        self.lbl_preprocess_recs = QLabel("")
        self.lbl_preprocess_recs.setWordWrap(True)
        self.lbl_preprocess_recs.setStyleSheet("font-size: 11px; color: #2b6cb0;")
        quality_layout.addWidget(self.lbl_preprocess_recs)

        # Toggle controls for auto-recommendations (hidden until recommendations exist)
        self._rec_toggle_widget = QWidget()
        _rec_toggle_layout = QHBoxLayout(self._rec_toggle_widget)
        _rec_toggle_layout.setContentsMargins(0, 0, 0, 0)
        self.chk_rec_date_extract = QCheckBox("Date/time features")
        self.chk_rec_date_extract.setVisible(False)
        _rec_toggle_layout.addWidget(self.chk_rec_date_extract)

        self.chk_rec_text_extract = QCheckBox("Text feature extraction")
        self.chk_rec_text_extract.setVisible(False)
        _rec_toggle_layout.addWidget(self.chk_rec_text_extract)

        self.chk_rec_rare_grouping = QCheckBox("Group rare categories")
        self.chk_rec_rare_grouping.setVisible(False)
        _rec_toggle_layout.addWidget(self.chk_rec_rare_grouping)

        _rec_toggle_layout.addStretch()
        quality_layout.addWidget(self._rec_toggle_widget)

        # Per-recommendation editors (hidden until recommendations exist)
        self._rec_edit_widget = QWidget()
        _rec_edit_layout = QHBoxLayout(self._rec_edit_widget)
        _rec_edit_layout.setContentsMargins(0, 0, 0, 0)

        # Text feature columns editor
        self.le_text_feature_columns = QLineEdit()
        self.le_text_feature_columns.setPlaceholderText("text columns (comma-separated)")
        self.le_text_feature_columns.setVisible(False)
        self.le_text_feature_columns.setFixedWidth(260)
        _rec_edit_layout.addWidget(self.le_text_feature_columns)

        # Rare-category min freq editor
        self.spin_rare_min_freq = QDoubleSpinBox()
        self.spin_rare_min_freq.setRange(0.0, 0.5)
        self.spin_rare_min_freq.setSingleStep(0.01)
        self.spin_rare_min_freq.setValue(0.05)
        self.spin_rare_min_freq.setSuffix(" min freq")
        self.spin_rare_min_freq.setVisible(False)
        _rec_edit_layout.addWidget(self.spin_rare_min_freq)

        _rec_edit_layout.addStretch()
        quality_layout.addWidget(self._rec_edit_widget)

        # Connect signals to refresh summary when user edits values
        try:
            self.chk_rec_date_extract.toggled.connect(lambda _: self._update_preprocess_summary())
            self.chk_rec_text_extract.toggled.connect(lambda _: self._update_preprocess_summary())
            self.chk_rec_rare_grouping.toggled.connect(lambda _: self._update_preprocess_summary())
            self.le_text_feature_columns.editingFinished.connect(self._update_preprocess_summary)
            self.spin_rare_min_freq.valueChanged.connect(lambda _: self._update_preprocess_summary())
        except Exception:
            pass
        # Persist preferences when user edits
        try:
            self.chk_rec_date_extract.toggled.connect(lambda _: self._save_preferences())
            self.chk_rec_text_extract.toggled.connect(lambda _: self._save_preferences())
            self.chk_rec_rare_grouping.toggled.connect(lambda _: self._save_preferences())
            self.le_text_feature_columns.editingFinished.connect(self._save_preferences)
            self.spin_rare_min_freq.valueChanged.connect(lambda _: self._save_preferences())
        except Exception:
            pass

        self.btn_export_quality_report = QPushButton("Export Quality Report")
        self.btn_export_quality_report.setEnabled(False)
        self.btn_export_quality_report.clicked.connect(self.on_export_quality_report_clicked)
        quality_layout.addWidget(self.btn_export_quality_report)

        quick_fix_row = QHBoxLayout()
        self.chk_drop_duplicate_rows = QCheckBox("Drop duplicate rows")
        self.chk_drop_duplicate_rows.setChecked(False)
        quick_fix_row.addWidget(self.chk_drop_duplicate_rows)

        self.chk_drop_constant_columns = QCheckBox("Drop constant columns")
        self.chk_drop_constant_columns.setChecked(True)
        quick_fix_row.addWidget(self.chk_drop_constant_columns)

        quick_fix_row.addWidget(QLabel("Missing values:"))
        self.combo_missing_strategy = QComboBox()
        self.combo_missing_strategy.addItem("None", "none")
        self.combo_missing_strategy.addItem("Drop rows", "drop_rows")
        self.combo_missing_strategy.addItem("Fill simple (median/mode)", "fill_simple")
        quick_fix_row.addWidget(self.combo_missing_strategy)
        quick_fix_row.addStretch()
        quality_layout.addLayout(quick_fix_row)

        self.btn_apply_quality_fixes = QPushButton("Apply Quick Fixes")
        self.btn_apply_quality_fixes.clicked.connect(self.on_apply_quality_fixes_clicked)
        quick_fix_action_row = QHBoxLayout()
        self.btn_preview_quality_fixes = QPushButton("Preview Quick Fixes")
        self.btn_preview_quality_fixes.clicked.connect(self.on_preview_quality_fixes_clicked)
        quick_fix_action_row.addWidget(self.btn_preview_quality_fixes)
        # Button to apply recommended automatic quick-fixes
        self.btn_apply_recommended = QPushButton("Apply Recommended Fixes")
        self.btn_apply_recommended.clicked.connect(self.on_apply_recommended_clicked)
        quick_fix_action_row.addWidget(self.btn_apply_recommended)
        quick_fix_action_row.addWidget(self.btn_apply_quality_fixes)

        self.btn_revert_quality_fixes = QPushButton("Revert Quick Fixes")
        self.btn_revert_quality_fixes.setEnabled(False)
        self.btn_revert_quality_fixes.clicked.connect(self.on_revert_quality_fixes_clicked)
        quick_fix_action_row.addWidget(self.btn_revert_quality_fixes)
        quality_layout.addLayout(quick_fix_action_row)
        quality_group.setLayout(quality_layout)
        top_layout.addWidget(quality_group)

        prep_group = QGroupBox("Preprocessing")
        prep_layout = QVBoxLayout()
        self.scale_checkbox = QCheckBox("Scale numeric features")
        self.scale_checkbox.setChecked(True)
        prep_layout.addWidget(self.scale_checkbox)

        self.encode_checkbox = QCheckBox("One-hot encode categorical features")
        self.encode_checkbox.setChecked(True)
        prep_layout.addWidget(self.encode_checkbox)

        test_row = QHBoxLayout()
        test_row.addWidget(QLabel("Test size:"))
        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setRange(0.1, 0.4)
        self.test_size_spin.setSingleStep(0.05)
        self.test_size_spin.setValue(0.2)
        self.test_size_label = QLabel("(20%)")
        self.test_size_spin.valueChanged.connect(
            lambda value: self.test_size_label.setText(f"({int(value * 100)}%)")
        )
        test_row.addWidget(self.test_size_spin)
        test_row.addWidget(self.test_size_label)
        test_row.addStretch()
        prep_layout.addLayout(test_row)

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Random seed:"))
        self.random_seed_spin = QSpinBox()
        self.random_seed_spin.setRange(0, 999999)
        self.random_seed_spin.setValue(42)
        seed_row.addWidget(self.random_seed_spin)
        seed_row.addStretch()
        prep_layout.addLayout(seed_row)

        prep_group.setLayout(prep_layout)
        top_layout.addWidget(prep_group)

        self.btn_run_train = QPushButton("Train + Evaluate")
        self.btn_run_train.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_run_train.clicked.connect(self.on_run_clicked)
        top_layout.addWidget(self.btn_run_train)

        self.lbl_training_status = QLabel("Ready")
        top_layout.addWidget(self.lbl_training_status)

        top_layout.addWidget(QLabel("Metrics:"))
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.setMaximumHeight(140)
        top_layout.addWidget(self.metrics_table)

        self.btn_save_model = QPushButton("Save Model")
        self.btn_save_model.setEnabled(False)
        self.btn_save_model.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_save_model.clicked.connect(self.on_save_model_clicked)
        top_layout.addWidget(self.btn_save_model)

        splitter.addWidget(top_widget)

        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        viz_layout.addWidget(QLabel("Evaluation Plots:"))

        self.viz_tabs = QTabWidget()
        self.viz_tabs.addTab(QLabel("Run training to see evaluation plots."), "Info")
        viz_layout.addWidget(self.viz_tabs)

        splitter.addWidget(viz_widget)
        splitter.setSizes([420, 480])

        self._refresh_model_options()

    def setup_predict_tab(self, tab: QWidget) -> None:
        layout = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Saved Models"))

        self.model_list = QListWidget()
        self.model_list.currentItemChanged.connect(self.on_model_selected)
        left_layout.addWidget(self.model_list)

        refresh_btn = QPushButton("Refresh List")
        refresh_btn.clicked.connect(self.refresh_model_list)
        left_layout.addWidget(refresh_btn)
        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.info_box = QGroupBox("Selected Model")
        info_layout = QVBoxLayout()
        self.lbl_model_info = QLabel("Select a model to view details.")
        self.lbl_model_info.setWordWrap(True)
        info_layout.addWidget(self.lbl_model_info)
        self.info_box.setLayout(info_layout)
        right_layout.addWidget(self.info_box)

        right_layout.addWidget(QLabel("Input Features"))
        self.predict_scroll = QScrollArea()
        self.predict_scroll.setWidgetResizable(True)

        self.predict_form_widget = QWidget()
        self.predict_form_layout = QFormLayout(self.predict_form_widget)
        self.predict_scroll.setWidget(self.predict_form_widget)
        right_layout.addWidget(self.predict_scroll)

        self.btn_predict = QPushButton("Predict")
        self.btn_predict.setEnabled(False)
        self.btn_predict.setStyleSheet("font-weight: bold; padding: 10px; background-color: #e0f7fa; color: black;")
        self.btn_predict.clicked.connect(self.on_predict_clicked)
        right_layout.addWidget(self.btn_predict)

        batch_row = QHBoxLayout()
        self.btn_load_batch = QPushButton("Load Batch CSV")
        self.btn_load_batch.clicked.connect(self.on_load_batch_csv_clicked)
        self.btn_batch_predict = QPushButton("Batch Predict + Export")
        self.btn_batch_predict.setEnabled(False)
        self.btn_batch_predict.clicked.connect(self.on_batch_predict_clicked)
        batch_row.addWidget(QLabel("Export profile:"))
        self.batch_output_profile_combo = QComboBox()
        self.batch_output_profile_combo.addItem("Detailed Output", "detailed")
        self.batch_output_profile_combo.addItem("Simple Output", "simple")
        self.batch_output_profile_combo.setToolTip(
            "Detailed includes original input columns and all prediction outputs.\n"
            "Simple includes only essential prediction columns."
        )
        batch_row.addWidget(self.btn_load_batch)
        batch_row.addWidget(self.btn_batch_predict)
        batch_row.addWidget(self.batch_output_profile_combo)
        right_layout.addLayout(batch_row)

        self.lbl_batch_status = QLabel("Batch input: not loaded")
        self.lbl_batch_status.setStyleSheet("font-size: 12px; color: #555;")
        right_layout.addWidget(self.lbl_batch_status)

        self.lbl_batch_preview = QLabel("Export columns preview: -")
        self.lbl_batch_preview.setWordWrap(True)
        self.lbl_batch_preview.setStyleSheet("font-size: 12px; color: #555;")
        right_layout.addWidget(self.lbl_batch_preview)

        self.lbl_prediction_warning = QLabel("")
        self.lbl_prediction_warning.setWordWrap(True)
        self.lbl_prediction_warning.setStyleSheet("color: #c56a00; font-size: 12px;")
        right_layout.addWidget(self.lbl_prediction_warning)

        self.lbl_prediction_result = QLabel("Result: -")
        self.lbl_prediction_result.setAlignment(Qt.AlignCenter)
        self.lbl_prediction_result.setStyleSheet("font-size: 16px; font-weight: bold; color: #00796b; padding: 5px;")
        right_layout.addWidget(self.lbl_prediction_result)

        splitter.addWidget(right_panel)
        splitter.setSizes([260, 640])

        self.refresh_model_list()

    def on_browse_clicked(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV",
            str(DATA_DIR),
            "CSV Files (*.csv)",
        )
        if not path_str:
            return

        self._load_csv(Path(path_str))

    def on_run_demo_clicked(self) -> None:
        try:
            demo_path = get_demo_dataset_path()
        except Exception as exc:
            QMessageBox.critical(self, "Demo Dataset Error", str(exc))
            return

        try:
            self._load_csv(demo_path)
            if self.target_combo.count() > 0:
                demo_target = "Survived"
                demo_index = self.target_combo.findText(demo_target)
                if demo_index >= 0:
                    self.target_combo.setCurrentIndex(demo_index)
            self.on_run_clicked()
        except Exception as exc:
            QMessageBox.critical(self, "Demo Run Error", str(exc))

    def _load_csv(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.training_csv_path = csv_path
        try:
            df = pd.read_csv(self.csv_path)
            if df.empty:
                raise ValueError("CSV has no rows")

            self.current_df = df
            self.original_df = df.copy(deep=True)
            self.file_label.setText(f"{self.csv_path.name} ({df.shape[0]} rows, {df.shape[1]} cols)")
            self.btn_revert_quality_fixes.setEnabled(False)
            self._update_training_source_label(is_cleaned=False)

            self.target_combo.clear()
            for col in df.columns:
                self.target_combo.addItem(col)
            self.target_combo.setEnabled(True)
            if self.target_combo.count() > 0:
                self.target_combo.setCurrentIndex(0)
            else:
                self._update_data_quality_labels(None)

            self.btn_save_model.setEnabled(False)
            self.last_run_context = None
            self.metrics_table.setRowCount(0)
            self.viz_tabs.clear()
            self.viz_tabs.addTab(QLabel("Run training to see evaluation plots."), "Info")
        except Exception as exc:
            QMessageBox.critical(self, "CSV Error", str(exc))

    def on_target_changed(self, index: int) -> None:
        if index >= 0:
            self.target_column = self.target_combo.itemText(index)
        self._refresh_data_quality()

    def _refresh_data_quality(self) -> None:
        if self.current_df is None:
            self.data_quality_report = None
            self._update_data_quality_labels(None)
            return

        try:
            self.data_quality_report = analyze_data_quality(self.current_df, self.target_column)
            # Also compute automated recommendations for quick-fixes
            try:
                self._auto_recommendations = recommend_quick_fixes(self.data_quality_report)
                rec_params = self._auto_recommendations.get("preprocess_params") or {}
                self._auto_preprocess_recommendations = rec_params if isinstance(rec_params, dict) else {}
            except Exception:
                self._auto_recommendations = None
                self._auto_preprocess_recommendations = {}
            self._update_data_quality_labels(self.data_quality_report)
        except Exception as exc:
            self.data_quality_report = None
            self._auto_recommendations = None
            self._auto_preprocess_recommendations = {}
            self.lbl_data_quality_summary.setText("Unable to compute quality summary.")
            self.lbl_data_quality_warnings.setText(str(exc))

    def on_apply_quality_fixes_clicked(self) -> None:
        if self.current_df is None:
            QMessageBox.warning(self, "Quick Fixes", "Please load a dataset first.")
            return

        try:
            fixed_df, preview_text, actions = self._build_quick_fix_preview()
            confirm_text = f"{preview_text}\n\nApply these changes?"

            choice = QMessageBox.question(
                self,
                "Confirm Quick Fixes",
                confirm_text,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if choice != QMessageBox.Yes:
                self.lbl_training_status.setText("Quick fixes cancelled by user.")
                return

            if fixed_df.equals(self.current_df):
                QMessageBox.information(self, "Quick Fixes", "No changes were needed.")
                return

            self.current_df = fixed_df

            base_name = self.csv_path.stem if self.csv_path else "dataset"
            cleaned_name = f"{base_name}_cleaned_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            cleaned_path = GENERATED_DIR / cleaned_name
            cleaned_path.parent.mkdir(parents=True, exist_ok=True)
            self.current_df.to_csv(cleaned_path, index=False)
            self.training_csv_path = cleaned_path
            self._update_training_source_label(is_cleaned=True)

            self.file_label.setText(
                f"{cleaned_path.name} ({self.current_df.shape[0]} rows, {self.current_df.shape[1]} cols)"
            )

            previous_target = self.target_column
            self.target_combo.blockSignals(True)
            self.target_combo.clear()
            for col in self.current_df.columns:
                self.target_combo.addItem(col)
            self.target_combo.blockSignals(False)

            if previous_target and previous_target in self.current_df.columns:
                idx = self.target_combo.findText(previous_target)
                if idx >= 0:
                    self.target_combo.setCurrentIndex(idx)
                self.target_column = previous_target
            elif self.target_combo.count() > 0:
                self.target_combo.setCurrentIndex(0)
                self.target_column = self.target_combo.currentText()

            self._refresh_data_quality()

            action_text = "\n".join(actions) if actions else "Applied selected quick fixes."
            self.lbl_training_status.setText("Quick fixes applied; training will use cleaned dataset.")
            self.btn_revert_quality_fixes.setEnabled(True)
            QMessageBox.information(self, "Quick Fixes Applied", action_text)
        except Exception as exc:
            QMessageBox.critical(self, "Quick Fixes Error", str(exc))

    def _build_quick_fix_preview(self) -> tuple[pd.DataFrame, str, List[str]]:
        if self.current_df is None:
            raise ValueError("Please load a dataset first.")

        strategy = str(self.combo_missing_strategy.currentData() or "none")
        before_rows, before_cols = self.current_df.shape
        fixed_df, actions = apply_quick_fixes(
            self.current_df,
            target_column=self.target_column,
            drop_constant_columns=self.chk_drop_constant_columns.isChecked(),
            drop_duplicate_rows=self.chk_drop_duplicate_rows.isChecked(),
            missing_strategy=strategy,
        )

        after_rows, after_cols = fixed_df.shape
        action_lines = actions if actions else ["No direct data changes detected from selected options."]
        preview_text = (
            "Quick-fix preview:\n"
            f"Rows: {before_rows} -> {after_rows}\n"
            f"Columns: {before_cols} -> {after_cols}\n\n"
            "Planned actions:\n- "
            + "\n- ".join(action_lines)
        )
        return fixed_df, preview_text, actions

    def on_preview_quality_fixes_clicked(self) -> None:
        if self.current_df is None:
            QMessageBox.warning(self, "Quick Fix Preview", "Please load a dataset first.")
            return

        try:
            _, preview_text, _ = self._build_quick_fix_preview()
            QMessageBox.information(self, "Quick Fix Preview", preview_text)
        except Exception as exc:
            QMessageBox.critical(self, "Quick Fix Preview Error", str(exc))

    def on_revert_quality_fixes_clicked(self) -> None:
        if self.csv_path is None or self.original_df is None:
            QMessageBox.warning(self, "Revert Quick Fixes", "No original dataset is available to restore.")
            return

        if self.current_df is not None and self.current_df.equals(self.original_df):
            QMessageBox.information(self, "Revert Quick Fixes", "Dataset is already in original state.")
            self.btn_revert_quality_fixes.setEnabled(False)
            return

        self.current_df = self.original_df.copy(deep=True)
        self.training_csv_path = self.csv_path
        self._update_training_source_label(is_cleaned=False)

        self.file_label.setText(
            f"{self.csv_path.name} ({self.current_df.shape[0]} rows, {self.current_df.shape[1]} cols)"
        )

        previous_target = self.target_column
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        for col in self.current_df.columns:
            self.target_combo.addItem(col)
        self.target_combo.blockSignals(False)

        if previous_target and previous_target in self.current_df.columns:
            idx = self.target_combo.findText(previous_target)
            if idx >= 0:
                self.target_combo.setCurrentIndex(idx)
            self.target_column = previous_target
        elif self.target_combo.count() > 0:
            self.target_combo.setCurrentIndex(0)
            self.target_column = self.target_combo.currentText()

        self._refresh_data_quality()
        self.lbl_training_status.setText("Reverted to original dataset; training will use original CSV.")
        self.btn_revert_quality_fixes.setEnabled(False)
        QMessageBox.information(self, "Revert Quick Fixes", "Restored original dataset state.")

    def _update_training_source_label(self, *, is_cleaned: bool) -> None:
        if self.training_csv_path is None:
            self.lbl_training_source.setText("Training source: -")
            return

        source_kind = "cleaned dataset" if is_cleaned else "original dataset"
        self.lbl_training_source.setText(
            f"Training source: {self.training_csv_path.name} ({source_kind})"
        )

    def _update_data_quality_labels(self, report: Dict[str, Any] | None) -> None:
        if not report:
            self.lbl_data_quality_summary.setText("Load a dataset to see quality summary.")
            self.lbl_data_quality_warnings.setText("")
            self.btn_export_quality_report.setEnabled(False)
            self.lbl_preprocess_recs.setText("")
            return

        summary = report.get("summary") or {}
        missing_columns = summary.get("missing_columns") or {}
        potential_id_columns = summary.get("potential_id_columns") or []

        self.lbl_data_quality_summary.setText(
            f"Rows: {summary.get('rows', 0)} | Columns: {summary.get('columns', 0)} | "
            f"Columns with missing values: {len(missing_columns)} | "
            f"Potential ID columns: {len(potential_id_columns)}"
        )

        warnings = report.get("warnings") or []
        if warnings:
            preview = " | ".join(str(item) for item in warnings[:3])
            extra = len(warnings) - 3
            if extra > 0:
                preview += f" | ... (+{extra} more)"
            self.lbl_data_quality_warnings.setText(f"Warnings: {preview}")
        else:
            self.lbl_data_quality_warnings.setText("No major quality warnings detected.")

        self.btn_export_quality_report.setEnabled(True)
        # Update preprocess recommendations hint
        recs = getattr(self, "_auto_preprocess_recommendations", {}) or {}
        if recs:
            parts: List[str] = []
            if recs.get("date_extract"):
                parts.append("extract date/time features")
                # show toggle and default to checked
                try:
                    self.chk_rec_date_extract.setVisible(True)
                    self.chk_rec_date_extract.setChecked(True)
                    # show date detail editors if desired (kept hidden for now)
                    self._rec_edit_widget.setVisible(True)
                except Exception:
                    pass
            if recs.get("text_extract"):
                cols = recs.get("text_feature_columns") or []
                if isinstance(cols, list) and cols:
                    parts.append(f"text features for {cols}")
                else:
                    parts.append("text feature extraction")
                try:
                    self.chk_rec_text_extract.setVisible(True)
                    self.chk_rec_text_extract.setChecked(True)
                    # prefill text column editor when available
                    if isinstance(cols, list) and cols:
                        self.le_text_feature_columns.setText(
                            ",".join(str(c) for c in cols)
                        )
                    self.le_text_feature_columns.setVisible(True)
                except Exception:
                    pass
            if recs.get("rare_category_min_freq"):
                parts.append("group rare categorical values")
                try:
                    self.chk_rec_rare_grouping.setVisible(True)
                    self.chk_rec_rare_grouping.setChecked(True)
                    try:
                        self.spin_rare_min_freq.setValue(float(recs.get("rare_category_min_freq", 0.05)))
                    except Exception:
                        pass
                    self.spin_rare_min_freq.setVisible(True)
                except Exception:
                    pass

            self.lbl_preprocess_recs.setText("Recommended preprocess: " + ", ".join(parts))
        else:
            self.lbl_preprocess_recs.setText("")
            # hide toggles when no recommendations
            try:
                self.chk_rec_date_extract.setVisible(False)
                self.chk_rec_text_extract.setVisible(False)
                self.chk_rec_rare_grouping.setVisible(False)
                self._rec_edit_widget.setVisible(False)
                self.le_text_feature_columns.setVisible(False)
                self.spin_rare_min_freq.setVisible(False)
            except Exception:
                pass

    def _update_preprocess_summary(self) -> None:
        """Refresh the `lbl_preprocess_recs` text to reflect current user edits."""
        recs = getattr(self, "_auto_preprocess_recommendations", {}) or {}
        if not recs:
            self.lbl_preprocess_recs.setText("")
            return

        parts: List[str] = []
        if recs.get("date_extract") and getattr(self, "chk_rec_date_extract", None) and self.chk_rec_date_extract.isChecked():
            parts.append("extract date/time features")
        if recs.get("text_extract") and getattr(self, "chk_rec_text_extract", None) and self.chk_rec_text_extract.isChecked():
            text_cols = self.le_text_feature_columns.text().strip() if getattr(self, "le_text_feature_columns", None) else ""
            if text_cols:
                parts.append(f"text features for [{text_cols}]")
            else:
                parts.append("text feature extraction")
        if recs.get("rare_category_min_freq") and getattr(self, "chk_rec_rare_grouping", None) and self.chec... (truncated)