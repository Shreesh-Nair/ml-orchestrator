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
from core.data_quality import analyze_data_quality, apply_quick_fixes, write_data_quality_report
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
        quality_layout.addWidget(self.btn_apply_quality_fixes)
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
            self.file_label.setText(f"{self.csv_path.name} ({df.shape[0]} rows, {df.shape[1]} cols)")

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
            self._update_data_quality_labels(self.data_quality_report)
        except Exception as exc:
            self.data_quality_report = None
            self.lbl_data_quality_summary.setText("Unable to compute quality summary.")
            self.lbl_data_quality_warnings.setText(str(exc))

    def on_apply_quality_fixes_clicked(self) -> None:
        if self.current_df is None:
            QMessageBox.warning(self, "Quick Fixes", "Please load a dataset first.")
            return

        try:
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
                + "\n\nApply these changes?"
            )

            choice = QMessageBox.question(
                self,
                "Confirm Quick Fixes",
                preview_text,
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
            QMessageBox.information(self, "Quick Fixes Applied", action_text)
        except Exception as exc:
            QMessageBox.critical(self, "Quick Fixes Error", str(exc))

    def _update_data_quality_labels(self, report: Dict[str, Any] | None) -> None:
        if not report:
            self.lbl_data_quality_summary.setText("Load a dataset to see quality summary.")
            self.lbl_data_quality_warnings.setText("")
            self.btn_export_quality_report.setEnabled(False)
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

    def on_export_quality_report_clicked(self) -> None:
        if not self.data_quality_report:
            QMessageBox.warning(
                self,
                "Export Quality Report",
                "Load a dataset to generate a data quality report first.",
            )
            return

        csv_name = self.csv_path.stem if self.csv_path else "dataset"
        default_name = f"{csv_name}_quality_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Data Quality Report",
            str(EXPORTS_DIR / default_name),
            "JSON Files (*.json)",
        )
        if not path_str:
            return

        out_path = Path(path_str)
        try:
            write_data_quality_report(
                self.data_quality_report,
                out_path,
                source_csv=str(self.csv_path) if self.csv_path else None,
                target_column=self.target_column,
            )
            QMessageBox.information(self, "Export Complete", f"Saved quality report to:\n{out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def on_task_changed(self, index: int) -> None:
        task = self.task_combo.itemData(index)
        if isinstance(task, str) and task in TASK_CONFIG:
            self.selected_task = task
        self._refresh_model_options()

    def on_model_changed(self, index: int) -> None:
        model_type = self.model_combo.itemData(index)
        if isinstance(model_type, str):
            self.selected_model_type = model_type

    def _refresh_model_options(self) -> None:
        task_cfg = TASK_CONFIG.get(self.selected_task, TASK_CONFIG["classification"])
        model_options = task_cfg["models"]

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for label, model_type in model_options:
            self.model_combo.addItem(label, model_type)
        self.model_combo.blockSignals(False)

        if self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(0)
            selected = self.model_combo.currentData()
            if isinstance(selected, str):
                self.selected_model_type = selected

    def _validate_target_for_task(self) -> Dict[str, int]:
        if self.current_df is None:
            raise ValueError("Please select a CSV file first.")
        if not self.target_column or self.target_column not in self.current_df.columns:
            raise ValueError("Please select a valid target column.")

        target = self.current_df[self.target_column].dropna()
        if target.empty:
            raise ValueError("Target column is empty after removing missing values.")

        class_counts = target.value_counts(dropna=False)
        task_cfg = TASK_CONFIG.get(self.selected_task, TASK_CONFIG["classification"])
        if bool(task_cfg.get("require_binary_target", False)):
            if len(class_counts) != 2:
                raise ValueError(
                    f"{task_cfg['label']} requires exactly 2 target classes. "
                    f"Found {len(class_counts)} classes: {list(class_counts.index)}"
                )
            if int(class_counts.min()) < 2:
                raise ValueError(
                    "Each class must have at least 2 rows for train/test split. "
                    f"Class counts: {class_counts.to_dict()}"
                )
        return class_counts.to_dict()

    def _validate_binary_target(self) -> Dict[str, int]:
        return self._validate_target_for_task()

    def on_run_clicked(self) -> None:
        if self.run_worker is not None and self.run_worker.isRunning():
            return

        if not self.csv_path or not self.target_column:
            QMessageBox.warning(self, "Inputs Missing", "Please select CSV and target column.")
            return

        try:
            self._validate_target_for_task()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Target", str(exc))
            return

        self.btn_save_model.setEnabled(False)
        self.last_run_context = None

        yaml_path = self._write_generated_yaml()

        self._set_training_state(True)
        self.run_worker = PipelineRunWorker(str(yaml_path.resolve()))
        self.run_worker.succeeded.connect(self._on_train_succeeded)
        self.run_worker.failed.connect(self._on_train_failed)
        self.run_worker.finished.connect(self._on_train_finished)
        self.run_worker.start()

    def _set_training_state(self, in_progress: bool) -> None:
        self.btn_run_train.setEnabled(not in_progress)
        self.btn_browse_csv.setEnabled(not in_progress)
        self.btn_run_demo.setEnabled(not in_progress)
        self.target_combo.setEnabled((not in_progress) and self.current_df is not None)
        self.scale_checkbox.setEnabled(not in_progress)
        self.encode_checkbox.setEnabled(not in_progress)
        self.test_size_spin.setEnabled(not in_progress)
        self.random_seed_spin.setEnabled(not in_progress)
        self.btn_save_session.setEnabled(not in_progress)
        self.btn_load_session.setEnabled(not in_progress)

        if in_progress:
            self.lbl_training_status.setText("Training in progress...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _on_train_succeeded(self, context: Dict[str, Any]) -> None:
        self.last_run_context = context
        self.btn_save_model.setEnabled(True)
        self._show_metrics(context)
        self._update_visualizations(context)
        task_label = TASK_CONFIG.get(self.selected_task, {}).get("label", "Task")
        self.lbl_training_status.setText(f"Training completed ({task_label})")

    def _on_train_failed(self, message: str) -> None:
        self.lbl_training_status.setText("Training failed")
        QMessageBox.critical(self, "Pipeline Error", message)

    def _on_train_finished(self) -> None:
        self._set_training_state(False)
        self.run_worker = None

    def _format_metric(self, value: Any) -> str:
        if isinstance(value, (int, float, np.floating, np.integer)):
            if isinstance(value, float) and math.isnan(value):
                return "nan"
            return f"{float(value):.4f}"
        return str(value)

    def _show_metrics(self, context: Dict[str, Any]) -> None:
        metrics = context.get("metrics") or {}
        if self.selected_task == "regression":
            preferred_order = ["rmse", "r2"]
        elif self.selected_task == "anomaly":
            metrics = context.get("anomaly_metrics") or metrics
            preferred_order = ["auc", "precision", "recall", "f1"]
        else:
            preferred_order = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        ordered_keys: List[str] = [key for key in preferred_order if key in metrics]
        ordered_keys.extend([key for key in metrics.keys() if key not in ordered_keys])

        self.metrics_table.setRowCount(0)
        for row, key in enumerate(ordered_keys):
            self.metrics_table.insertRow(row)
            self.metrics_table.setItem(row, 0, QTableWidgetItem(key))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(self._format_metric(metrics[key])))

    def _update_visualizations(self, context: Dict[str, Any]) -> None:
        self.viz_tabs.clear()
        artifacts = context.get("artifacts") or {}

        if not artifacts:
            self.viz_tabs.addTab(QLabel("No visualization data available."), "Info")
            return

        if self.selected_task == "regression":
            if "y_test" in artifacts and "y_pred" in artifacts:
                y_test = np.asarray(artifacts["y_test"])
                y_pred = np.asarray(artifacts["y_pred"])

                canvas = MplCanvas(self, width=5, height=4, dpi=100)
                ax = canvas.axes
                ax.scatter(y_test, y_pred, alpha=0.7)

                diagonal_min = float(min(np.min(y_test), np.min(y_pred)))
                diagonal_max = float(max(np.max(y_test), np.max(y_pred)))
                ax.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], "r--", linewidth=1.5)

                ax.set_xlabel("Actual", fontsize=9)
                ax.set_ylabel("Predicted", fontsize=9)
                ax.set_title("Actual vs Predicted", fontsize=10)
                canvas.figure.tight_layout()
                self.viz_tabs.addTab(canvas, "Predictions")

            if "feature_importance" in artifacts:
                importances = np.asarray(artifacts["feature_importance"])
                if importances.ndim == 1 and len(importances) > 0:
                    names = artifacts.get("feature_names", [f"feature_{i}" for i in range(len(importances))])
                    sorted_idx = np.argsort(importances)[::-1][:10]

                    canvas = MplCanvas(self, width=5, height=4, dpi=100)
                    ax = canvas.axes
                    ax.bar(range(len(sorted_idx)), importances[sorted_idx], align="center")
                    ax.set_xticks(range(len(sorted_idx)))

                    clean_names: List[str] = []
                    for idx in sorted_idx:
                        label = str(names[idx]).split("__")[-1]
                        if len(label) > 20:
                            label = f"{label[:17]}..."
                        clean_names.append(label)

                    ax.set_xticklabels(clean_names, rotation=45, ha="right", fontsize=9)
                    ax.set_title("Top Feature Importances", fontsize=10)
                    canvas.figure.tight_layout()
                    self.viz_tabs.addTab(canvas, "Feature Importance")

            if self.viz_tabs.count() == 0:
                self.viz_tabs.addTab(QLabel("No visualization data available for regression."), "Info")
            return

        if self.selected_task == "anomaly":
            if "anomaly_scores" in artifacts:
                scores = np.asarray(artifacts["anomaly_scores"])

                canvas = MplCanvas(self, width=5, height=4, dpi=100)
                ax = canvas.axes
                ax.hist(scores, bins=min(30, max(5, int(np.sqrt(len(scores))))), color="#4C78A8", alpha=0.85)
                ax.set_xlabel("Anomaly Score", fontsize=9)
                ax.set_ylabel("Count", fontsize=9)
                ax.set_title("Anomaly Score Distribution", fontsize=10)
                canvas.figure.tight_layout()
                self.viz_tabs.addTab(canvas, "Score Distribution")

            if "y_test" in artifacts and "anomaly_preds" in artifacts:
                from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

                y_test = np.asarray(artifacts["y_test"])
                preds = np.asarray(artifacts["anomaly_preds"])
                labels = np.asarray([0, 1])
                cm = confusion_matrix(y_test, preds, labels=labels)

                canvas = MplCanvas(self, width=5, height=4, dpi=100)
                display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
                display.plot(ax=canvas.axes, cmap="Oranges", colorbar=True)
                canvas.axes.set_title("Anomaly Detection Confusion Matrix", fontsize=10)
                canvas.figure.tight_layout()
                self.viz_tabs.addTab(canvas, "Confusion Matrix")

            if self.viz_tabs.count() == 0:
                self.viz_tabs.addTab(QLabel("No visualization data available for anomaly detection."), "Info")
            return

        if "y_test" in artifacts and "y_pred" in artifacts:
            from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

            y_test = np.asarray(artifacts["y_test"])
            y_pred = np.asarray(artifacts["y_pred"])
            labels = np.asarray(artifacts.get("classes", sorted(np.unique(y_test))))
            cm = confusion_matrix(y_test, y_pred, labels=labels)

            canvas = MplCanvas(self, width=5, height=4, dpi=100)
            display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            display.plot(ax=canvas.axes, cmap="Blues", colorbar=True)
            canvas.axes.set_title("Confusion Matrix", fontsize=10)
            canvas.figure.tight_layout()
            self.viz_tabs.addTab(canvas, "Confusion Matrix")

        if "feature_importance" in artifacts:
            importances = np.asarray(artifacts["feature_importance"])
            if importances.ndim == 1 and len(importances) > 0:
                names = artifacts.get("feature_names", [f"feature_{i}" for i in range(len(importances))])
                sorted_idx = np.argsort(importances)[::-1][:10]

                canvas = MplCanvas(self, width=5, height=4, dpi=100)
                ax = canvas.axes
                ax.bar(range(len(sorted_idx)), importances[sorted_idx], align="center")
                ax.set_xticks(range(len(sorted_idx)))

                clean_names: List[str] = []
                for idx in sorted_idx:
                    label = str(names[idx]).split("__")[-1]
                    if len(label) > 20:
                        label = f"{label[:17]}..."
                    clean_names.append(label)

                ax.set_xticklabels(clean_names, rotation=45, ha="right", fontsize=9)
                ax.set_title("Top Feature Importances", fontsize=10)
                canvas.figure.tight_layout()
                self.viz_tabs.addTab(canvas, "Feature Importance")

        if "y_proba" in artifacts and "y_test" in artifacts:
            from sklearn.metrics import auc, roc_curve

            classes = list(np.asarray(artifacts.get("classes", [])))
            positive_label = artifacts.get("positive_label")
            if len(classes) == 2:
                if positive_label not in classes:
                    positive_label = classes[1]

                y_test = np.asarray(artifacts["y_test"])
                y_proba = np.asarray(artifacts["y_proba"])
                positive_index = classes.index(positive_label)
                y_true_bin = (y_test == positive_label).astype(int)

                if len(np.unique(y_true_bin)) == 2:
                    fpr, tpr, _ = roc_curve(y_true_bin, y_proba[:, positive_index])
                    roc_auc = auc(fpr, tpr)

                    canvas = MplCanvas(self, width=5, height=4, dpi=100)
                    ax = canvas.axes
                    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
                    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel("False Positive Rate", fontsize=9)
                    ax.set_ylabel("True Positive Rate", fontsize=9)
                    ax.set_title(f"ROC Curve (Positive={positive_label})", fontsize=10)
                    ax.legend(loc="lower right", fontsize=9)
                    canvas.figure.tight_layout()
                    self.viz_tabs.addTab(canvas, "ROC Curve")

    def _write_generated_yaml(self) -> Path:
        import yaml

        task_cfg = TASK_CONFIG.get(self.selected_task, TASK_CONFIG["classification"])
        model_type = self.selected_model_type
        pipeline_name = f"{self.selected_task}_{model_type}_gui_run"

        config = {
            "pipeline_name": pipeline_name,
            "stages": [
                {
                    "name": "load_data",
                    "type": "csv_loader",
                    "params": {
                        "source": str(self.training_csv_path or self.csv_path),
                        "target_column": self.target_column,
                    },
                },
                {
                    "name": "preprocess",
                    "type": "tabular_preprocess",
                    "params": {
                        "target_column": self.target_column,
                        "task_type": self.selected_task,
                        "require_binary_target": bool(task_cfg.get("require_binary_target", False)),
                        "scale_numeric": self.scale_checkbox.isChecked(),
                        "encode_categoricals": self.encode_checkbox.isChecked(),
                        "test_size": float(self.test_size_spin.value()),
                        "random_state": int(self.random_seed_spin.value()),
                    },
                },
                {
                    "name": "model",
                    "type": model_type,
                    "params": {"random_state": int(self.random_seed_spin.value())},
                },
            ],
        }

        yaml_path = GENERATED_DIR / f"{pipeline_name}.yml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        return yaml_path

    def on_save_model_clicked(self) -> None:
        if not self.last_run_context:
            QMessageBox.warning(self, "Nothing to Save", "Please train a model first.")
            return

        context = self.last_run_context
        task_cfg = TASK_CONFIG.get(self.selected_task, TASK_CONFIG["classification"])
        model = context.get("model")
        if model is None and self.selected_task == "anomaly":
            model = context.get("anomaly_model")
        preprocessor = context.get("preprocessor")
        if model is None:
            QMessageBox.warning(self, "Save Error", "No trained model found in the last run.")
            return

        artifacts = context.get("artifacts") or {}
        class_labels = context.get("class_labels") or list(np.asarray(artifacts.get("classes", [])))
        feature_columns = context.get("feature_columns") or []
        feature_dtypes = context.get("feature_dtypes") or {}
        feature_allowed_values: Dict[str, List[str]] = {}

        if not feature_columns and self.current_df is not None and self.target_column in self.current_df.columns:
            feature_columns = list(self.current_df.columns.drop(self.target_column))
            feature_dtypes = {col: str(self.current_df[col].dtype) for col in feature_columns}

        if self.current_df is not None:
            for feature in feature_columns:
                if feature not in self.current_df.columns:
                    continue
                dtype_text = str(feature_dtypes.get(feature, "")).lower()
                if not any(token in dtype_text for token in ["object", "category", "bool"]):
                    continue

                series = self.current_df[feature].dropna()
                if series.empty:
                    continue

                # Limit to most frequent categories to keep the UI compact and useful.
                values = series.astype(str).value_counts().index.tolist()[:25]
                if values:
                    feature_allowed_values[feature] = values

        default_name = f"{self.selected_task}_{self.selected_model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model",
            str(MODELS_DIR / default_name),
            "Model Files (*.pkl)",
        )
        if not path_str:
            return

        meta = {
            "task": self.selected_task,
            "algorithm": self.model_combo.currentText() if self.model_combo.currentText() else self.selected_model_type,
            "model_type": self.selected_model_type,
            "dataset": self.csv_path.name if self.csv_path else "unknown",
            "target": self.target_column,
            "feature_columns": feature_columns,
            "feature_dtypes": feature_dtypes,
            "feature_allowed_values": feature_allowed_values,
            "class_labels": [str(label) if isinstance(label, Path) else label for label in class_labels],
            "positive_label": artifacts.get("positive_label"),
            "preprocess": {
                "task": task_cfg.get("label", self.selected_task),
                "scale_numeric": self.scale_checkbox.isChecked(),
                "encode_categoricals": self.encode_checkbox.isChecked(),
                "test_size": float(self.test_size_spin.value()),
                "random_seed": int(self.random_seed_spin.value()),
            },
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": context.get("metrics", {}),
        }

        payload = {
            "meta": meta,
            "objects": {
                "model": model,
                "preprocessor": preprocessor,
            },
        }

        try:
            joblib.dump(payload, path_str)
            self.refresh_model_list()
            self.tabs.setCurrentIndex(1)
            QMessageBox.information(self, "Saved", "Model saved.")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def refresh_model_list(self) -> None:
        self.model_list.clear()
        if not MODELS_DIR.exists():
            return

        files = sorted(MODELS_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in files:
            self.model_list.addItem(path.name)

    def on_model_selected(self, item) -> None:
        if item is None:
            return

        model_path = MODELS_DIR / item.text()
        self.btn_predict.setEnabled(False)
        self.btn_batch_predict.setEnabled(False)
        self.loaded_model_payload = None
        self.prediction_schema = []
        self.prediction_allowed_values = {}
        self.batch_input_df = None
        self.batch_input_path = None
        self.lbl_prediction_result.setText("Result: -")
        self.lbl_prediction_warning.setText("")
        self.lbl_batch_status.setText("Batch input: not loaded")
        self.lbl_batch_preview.setText("Export columns preview: -")

        try:
            payload = joblib.load(model_path)
            if not isinstance(payload, dict) or "meta" not in payload or "objects" not in payload:
                raise ValueError("Invalid model file format")

            meta = payload["meta"]
            model_task = str(meta.get("task", "binary_classification"))
            if model_task == "binary_classification":
                model_task = "classification"
            if model_task not in TASK_CONFIG:
                raise ValueError(f"Unsupported model task: {model_task}")

            feature_columns = meta.get("feature_columns")
            feature_dtypes = meta.get("feature_dtypes", {})
            raw_allowed_values = meta.get("feature_allowed_values", {})
            if not isinstance(feature_columns, list) or not feature_columns:
                raise ValueError("Model file missing feature schema")

            if isinstance(raw_allowed_values, dict):
                self.prediction_allowed_values = {
                    str(k): [str(v) for v in values] for k, values in raw_allowed_values.items() if isinstance(values, list)
                }

            self.prediction_schema = [
                {"name": feature, "dtype": str(feature_dtypes.get(feature, "object"))}
                for feature in feature_columns
            ]

            metrics = meta.get("metrics", {})
            metrics_text = ", ".join([f"{k}={self._format_metric(v)}" for k, v in metrics.items()])
            classes_text = ", ".join([str(c) for c in meta.get("class_labels", [])])

            self.lbl_model_info.setText(
                f"Task: {model_task}\n"
                f"Algorithm: {meta.get('algorithm')}\n"
                f"Dataset: {meta.get('dataset')}\n"
                f"Target: {meta.get('target')}\n"
                f"Classes: {classes_text}\n"
                f"Date: {meta.get('date')}\n"
                f"Metrics: {metrics_text if metrics_text else '-'}"
            )

            self.loaded_model_payload = payload
            self._generate_prediction_form(self.prediction_schema)
            self._update_prediction_warnings()
            self.btn_predict.setEnabled(True)
            self.btn_batch_predict.setEnabled(self.batch_input_df is not None)
        except Exception as exc:
            self.lbl_model_info.setText(f"Error loading model: {exc}")

    def _generate_prediction_form(self, schema: List[Dict[str, str]]) -> None:
        while self.predict_form_layout.count():
            child = self.predict_form_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.prediction_inputs = {}
        for item in schema:
            feature = item["name"]
            dtype = item["dtype"]
            allowed_values = self.prediction_allowed_values.get(feature, [])

            if allowed_values:
                widget = QComboBox()
                widget.setEditable(True)
                widget.setInsertPolicy(QComboBox.NoInsert)
                widget.addItems(allowed_values)
                if widget.lineEdit() is not None:
                    widget.lineEdit().setPlaceholderText(f"Select or type {feature}")
                widget.currentTextChanged.connect(self._update_prediction_warnings)
            else:
                widget = QLineEdit()
                widget.setPlaceholderText(f"Enter {feature}")
                widget.textChanged.connect(self._update_prediction_warnings)

            self.predict_form_layout.addRow(f"{feature} ({dtype}):", widget)
            self.prediction_inputs[feature] = widget

    def _widget_text(self, widget: Any) -> str:
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QLineEdit):
            return widget.text()
        return ""

    def _update_prediction_warnings(self) -> None:
        warnings: List[str] = []

        for item in self.prediction_schema:
            feature = item["name"]
            widget = self.prediction_inputs.get(feature)
            if widget is None:
                continue

            allowed_values = self.prediction_allowed_values.get(feature, [])
            if not allowed_values:
                continue

            raw_value = self._widget_text(widget).strip()
            if not raw_value:
                continue

            allowed_norm = {v.strip().lower() for v in allowed_values}
            if raw_value.lower() not in allowed_norm:
                warnings.append(
                    f"{feature}: '{raw_value}' was not seen in training data. "
                    "Prediction may be less reliable."
                )

        if warnings:
            self.lbl_prediction_warning.setText("Input warning: " + " | ".join(warnings[:3]))
        else:
            self.lbl_prediction_warning.setText("")

    def _coerce_input_value(self, feature: str, raw_value: str, dtype: str) -> Any:
        value = raw_value.strip()
        if value == "":
            raise ValueError(f"Missing value for '{feature}'")

        normalized = dtype.lower()

        if "int" in normalized:
            float_val = float(value)
            if not float_val.is_integer():
                raise ValueError(f"'{feature}' expects an integer")
            return int(float_val)

        if any(token in normalized for token in ["float", "double", "decimal"]):
            return float(value)

        if "bool" in normalized:
            low = value.lower()
            if low in {"true", "1", "yes", "y"}:
                return True
            if low in {"false", "0", "no", "n"}:
                return False
            raise ValueError(f"'{feature}' expects a boolean (true/false)")

        return value

    def on_predict_clicked(self) -> None:
        if not self.loaded_model_payload:
            return

        objects = self.loaded_model_payload.get("objects", {})
        meta = self.loaded_model_payload.get("meta", {})
        model = objects.get("model")
        preprocessor = objects.get("preprocessor")
        model_task = str(meta.get("task", "classification"))
        if model_task == "binary_classification":
            model_task = "classification"

        if model is None:
            QMessageBox.critical(self, "Prediction Error", "Loaded file does not contain a model.")
            return

        try:
            row: Dict[str, Any] = {}
            for item in self.prediction_schema:
                feature = item["name"]
                dtype = item["dtype"]
                widget = self.prediction_inputs[feature]
                row[feature] = self._coerce_input_value(feature, self._widget_text(widget), dtype)

            input_df = pd.DataFrame([row])
            X = preprocessor.transform(input_df) if preprocessor is not None else input_df

            prediction = model.predict(X)[0]
            if model_task == "regression":
                result = f"Predicted value: {float(prediction):.4f}"
            elif model_task == "anomaly":
                anomaly_label = "Anomaly" if int(prediction) == -1 else "Normal"
                result = f"Predicted status: {anomaly_label}"
                if hasattr(model, "decision_function"):
                    score = -float(model.decision_function(X)[0])
                    result += f" | anomaly score = {score:.4f}"
            else:
                result = f"Predicted class: {prediction}"
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)[0]
                    classes = list(getattr(model, "classes_", meta.get("class_labels", [])))
                    positive_label = meta.get("positive_label")
                    if positive_label in classes:
                        positive_idx = classes.index(positive_label)
                        result += f" | P({positive_label}) = {float(probs[positive_idx]):.3f}"

            self.lbl_prediction_result.setText(result)
        except ValueError as exc:
            QMessageBox.warning(self, "Input Error", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Prediction Error", str(exc))

    def on_load_batch_csv_clicked(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select Batch CSV",
            str(DATA_DIR),
            "CSV Files (*.csv)",
        )
        if not path_str:
            return

        csv_path = Path(path_str)
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                raise ValueError("Batch CSV has no rows")

            self.batch_input_df = df
            self.batch_input_path = csv_path
            self.lbl_batch_status.setText(
                f"Batch input: {csv_path.name} ({df.shape[0]} rows, {df.shape[1]} cols)"
            )
            self.lbl_batch_preview.setText("Export columns preview: ready after prediction")
            self.btn_batch_predict.setEnabled(self.loaded_model_payload is not None)
        except Exception as exc:
            self.batch_input_df = None
            self.batch_input_path = None
            self.btn_batch_predict.setEnabled(False)
            QMessageBox.critical(self, "Batch CSV Error", str(exc))

    def on_batch_predict_clicked(self) -> None:
        if self.loaded_model_payload is None:
            QMessageBox.warning(self, "Batch Prediction", "Please load a saved model first.")
            return
        if self.batch_input_df is None:
            QMessageBox.warning(self, "Batch Prediction", "Please load a batch CSV first.")
            return

        try:
            output_profile = str(self.batch_output_profile_combo.currentData() or "detailed")
            output_df = predict_dataframe(
                self.loaded_model_payload,
                self.batch_input_df,
                output_profile=output_profile,
            )

            preview_cols = output_df.columns.tolist()
            preview_text = ", ".join(preview_cols[:8])
            if len(preview_cols) > 8:
                preview_text += f", ... (+{len(preview_cols) - 8} more)"
            self.lbl_batch_preview.setText(f"Export columns preview: {preview_text}")

            default_input = self.batch_input_path.stem if self.batch_input_path else "batch"
            default_name = f"{default_input}_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            out_path_str, _ = QFileDialog.getSaveFileName(
                self,
                "Save Batch Predictions",
                str(EXPORTS_DIR / default_name),
                "CSV Files (*.csv)",
            )
            if not out_path_str:
                return

            out_path = Path(out_path_str)
            output_df.to_csv(out_path, index=False)
            self.lbl_batch_status.setText(
                f"Batch output: {out_path.name} ({output_df.shape[0]} rows, profile={output_profile})"
            )
            QMessageBox.information(self, "Batch Prediction Complete", f"Saved predictions to:\n{out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Batch Prediction Error", str(exc))

    def _session_payload(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "saved_at": datetime.datetime.now().isoformat(),
            "csv_path": str(self.csv_path) if self.csv_path else None,
            "target_column": self.target_column,
            "preprocess": {
                "scale_numeric": self.scale_checkbox.isChecked(),
                "encode_categoricals": self.encode_checkbox.isChecked(),
                "test_size": float(self.test_size_spin.value()),
                "random_seed": int(self.random_seed_spin.value()),
            },
            "data_quality_fixes": {
                "drop_duplicate_rows": self.chk_drop_duplicate_rows.isChecked(),
                "drop_constant_columns": self.chk_drop_constant_columns.isChecked(),
                "missing_strategy": str(self.combo_missing_strategy.currentData() or "none"),
            },
        }

    def on_save_session_clicked(self) -> None:
        default_name = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Session",
            str((self.current_session_path or (PROJECTS_DIR / default_name)).resolve()),
            "Session Files (*.json)",
        )
        if not path_str:
            return

        session_path = Path(path_str)
        try:
            session_path.write_text(json.dumps(self._session_payload(), indent=2), encoding="utf-8")
            self.current_session_path = session_path
            QMessageBox.information(self, "Session Saved", f"Saved session to:\n{session_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Session Save Error", str(exc))

    def on_load_session_clicked(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load Session",
            str(PROJECTS_DIR),
            "Session Files (*.json)",
        )
        if not path_str:
            return

        session_path = Path(path_str)
        try:
            payload = json.loads(session_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Session file must contain a JSON object")

            preprocess = payload.get("preprocess") or {}
            self.scale_checkbox.setChecked(bool(preprocess.get("scale_numeric", True)))
            self.encode_checkbox.setChecked(bool(preprocess.get("encode_categoricals", True)))
            self.test_size_spin.setValue(float(preprocess.get("test_size", 0.2)))
            self.random_seed_spin.setValue(int(preprocess.get("random_seed", 42)))

            data_quality_fixes = payload.get("data_quality_fixes") or {}
            self.chk_drop_duplicate_rows.setChecked(bool(data_quality_fixes.get("drop_duplicate_rows", False)))
            self.chk_drop_constant_columns.setChecked(bool(data_quality_fixes.get("drop_constant_columns", True)))
            missing_strategy = str(data_quality_fixes.get("missing_strategy", "none"))
            idx_strategy = self.combo_missing_strategy.findData(missing_strategy)
            self.combo_missing_strategy.setCurrentIndex(idx_strategy if idx_strategy >= 0 else 0)

            csv_path = payload.get("csv_path")
            if isinstance(csv_path, str) and csv_path:
                csv_file = Path(csv_path)
                if not csv_file.exists():
                    raise FileNotFoundError(f"Session CSV not found: {csv_file}")
                self._load_csv(csv_file)

            target_column = payload.get("target_column")
            if isinstance(target_column, str) and self.current_df is not None:
                idx = self.target_combo.findText(target_column)
                if idx >= 0:
                    self.target_combo.setCurrentIndex(idx)

            self.current_session_path = session_path
            QMessageBox.information(self, "Session Loaded", f"Loaded session from:\n{session_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Session Load Error", str(exc))


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1120, 860)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
