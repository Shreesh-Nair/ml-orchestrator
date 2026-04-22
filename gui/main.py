from __future__ import annotations

import datetime
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
from core.paths import (
    get_data_dir,
    get_demo_dataset_path,
    get_generated_pipelines_dir,
    get_models_dir,
)

matplotlib.use("QtAgg")

DATA_DIR = get_data_dir()
GENERATED_DIR = get_generated_pipelines_dir()
MODELS_DIR = get_models_dir()


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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ML Orchestrator - Binary Classification Demo")

        self.csv_path: Path | None = None
        self.target_column: str | None = None
        self.current_df: pd.DataFrame | None = None

        self.last_run_context: Dict[str, Any] | None = None
        self.loaded_model_payload: Dict[str, Any] | None = None

        self.prediction_inputs: Dict[str, QLineEdit] = {}
        self.prediction_schema: List[Dict[str, str]] = []
        self.run_worker: PipelineRunWorker | None = None

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

        task_row = QHBoxLayout()
        task_row.addWidget(QLabel("Task:"))
        task_row.addWidget(QLabel("Binary Classification (Demo)"))
        task_row.addStretch()
        top_layout.addLayout(task_row)

        algo_row = QHBoxLayout()
        algo_row.addWidget(QLabel("Algorithm:"))
        algo_row.addWidget(QLabel("RandomForest (Demo)"))
        algo_row.addStretch()
        top_layout.addLayout(algo_row)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target column:"))
        self.target_combo = QComboBox()
        self.target_combo.setEnabled(False)
        self.target_combo.currentIndexChanged.connect(self.on_target_changed)
        target_row.addWidget(self.target_combo)
        top_layout.addLayout(target_row)

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

    def _validate_binary_target(self) -> Dict[str, int]:
        if self.current_df is None:
            raise ValueError("Please select a CSV file first.")
        if not self.target_column or self.target_column not in self.current_df.columns:
            raise ValueError("Please select a valid target column.")

        target = self.current_df[self.target_column].dropna()
        if target.empty:
            raise ValueError("Target column is empty after removing missing values.")

        class_counts = target.value_counts(dropna=False)
        if len(class_counts) != 2:
            raise ValueError(
                "Target must have exactly 2 classes for this demo. "
                f"Found {len(class_counts)} classes: {list(class_counts.index)}"
            )
        if int(class_counts.min()) < 2:
            raise ValueError(
                "Each class must have at least 2 rows for train/test split. "
                f"Class counts: {class_counts.to_dict()}"
            )
        return class_counts.to_dict()

    def on_run_clicked(self) -> None:
        if self.run_worker is not None and self.run_worker.isRunning():
            return

        if not self.csv_path or not self.target_column:
            QMessageBox.warning(self, "Inputs Missing", "Please select CSV and target column.")
            return

        try:
            self._validate_binary_target()
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
        self.lbl_training_status.setText("Training completed")

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

        config = {
            "pipeline_name": "binary_classification_gui_run",
            "stages": [
                {
                    "name": "load_data",
                    "type": "csv_loader",
                    "params": {
                        "source": str(self.csv_path),
                        "target_column": self.target_column,
                    },
                },
                {
                    "name": "preprocess",
                    "type": "tabular_preprocess",
                    "params": {
                        "target_column": self.target_column,
                        "task_type": "classification",
                        "require_binary_target": True,
                        "scale_numeric": self.scale_checkbox.isChecked(),
                        "encode_categoricals": self.encode_checkbox.isChecked(),
                        "test_size": float(self.test_size_spin.value()),
                    },
                },
                {
                    "name": "model",
                    "type": "classification_rf",
                    "params": {},
                },
            ],
        }

        yaml_path = GENERATED_DIR / "binary_classification_gui_run.yml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        return yaml_path

    def on_save_model_clicked(self) -> None:
        if not self.last_run_context:
            QMessageBox.warning(self, "Nothing to Save", "Please train a model first.")
            return

        context = self.last_run_context
        model = context.get("model")
        preprocessor = context.get("preprocessor")
        if model is None:
            QMessageBox.warning(self, "Save Error", "No trained model found in the last run.")
            return

        artifacts = context.get("artifacts") or {}
        class_labels = context.get("class_labels") or list(np.asarray(artifacts.get("classes", [])))
        feature_columns = context.get("feature_columns") or []
        feature_dtypes = context.get("feature_dtypes") or {}

        if not feature_columns and self.current_df is not None and self.target_column in self.current_df.columns:
            feature_columns = list(self.current_df.columns.drop(self.target_column))
            feature_dtypes = {col: str(self.current_df[col].dtype) for col in feature_columns}

        default_name = f"binary_rf_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model",
            str(MODELS_DIR / default_name),
            "Model Files (*.pkl)",
        )
        if not path_str:
            return

        meta = {
            "task": "binary_classification",
            "algorithm": "RandomForest",
            "dataset": self.csv_path.name if self.csv_path else "unknown",
            "target": self.target_column,
            "feature_columns": feature_columns,
            "feature_dtypes": feature_dtypes,
            "class_labels": [str(label) if isinstance(label, Path) else label for label in class_labels],
            "positive_label": artifacts.get("positive_label"),
            "preprocess": {
                "scale_numeric": self.scale_checkbox.isChecked(),
                "encode_categoricals": self.encode_checkbox.isChecked(),
                "test_size": float(self.test_size_spin.value()),
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
        self.loaded_model_payload = None
        self.prediction_schema = []
        self.lbl_prediction_result.setText("Result: -")

        try:
            payload = joblib.load(model_path)
            if not isinstance(payload, dict) or "meta" not in payload or "objects" not in payload:
                raise ValueError("Invalid model file format")

            meta = payload["meta"]
            if meta.get("task") != "binary_classification":
                raise ValueError("This model is not from the binary-classification demo flow")

            feature_columns = meta.get("feature_columns")
            feature_dtypes = meta.get("feature_dtypes", {})
            if not isinstance(feature_columns, list) or not feature_columns:
                raise ValueError("Model file missing feature schema")

            self.prediction_schema = [
                {"name": feature, "dtype": str(feature_dtypes.get(feature, "object"))}
                for feature in feature_columns
            ]

            metrics = meta.get("metrics", {})
            metrics_text = ", ".join([f"{k}={self._format_metric(v)}" for k, v in metrics.items()])
            classes_text = ", ".join([str(c) for c in meta.get("class_labels", [])])

            self.lbl_model_info.setText(
                f"Algorithm: {meta.get('algorithm')}\n"
                f"Dataset: {meta.get('dataset')}\n"
                f"Target: {meta.get('target')}\n"
                f"Classes: {classes_text}\n"
                f"Date: {meta.get('date')}\n"
                f"Metrics: {metrics_text if metrics_text else '-'}"
            )

            self.loaded_model_payload = payload
            self._generate_prediction_form(self.prediction_schema)
            self.btn_predict.setEnabled(True)
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
            widget = QLineEdit()
            widget.setPlaceholderText(f"Enter {feature}")
            self.predict_form_layout.addRow(f"{feature} ({dtype}):", widget)
            self.prediction_inputs[feature] = widget

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

        if model is None:
            QMessageBox.critical(self, "Prediction Error", "Loaded file does not contain a model.")
            return

        try:
            row: Dict[str, Any] = {}
            for item in self.prediction_schema:
                feature = item["name"]
                dtype = item["dtype"]
                widget = self.prediction_inputs[feature]
                row[feature] = self._coerce_input_value(feature, widget.text(), dtype)

            input_df = pd.DataFrame([row])
            X = preprocessor.transform(input_df) if preprocessor is not None else input_df

            prediction = model.predict(X)[0]
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


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1120, 860)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
