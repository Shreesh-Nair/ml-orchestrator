import sys
import os
import datetime
from pathlib import Path
from typing import Dict, Any, List
import joblib

# Make project root importable so `core` and `handlers` can be found
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QFileDialog,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QCheckBox,
    QDoubleSpinBox,
    QTabWidget,
    QScrollArea,
    QFormLayout,
    QLineEdit,
    QSplitter,
    QListWidget,
    QFrame,
    QGroupBox,
)
from PySide6.QtCore import Qt

from core.executor import run_pipeline

EXAMPLES_DIR = Path("examples")
GENERATED_DIR = EXAMPLES_DIR / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ML Orchestrator GUI")

        self.csv_path: Path | None = None
        self.target_column: str | None = None
        self.current_df: pd.DataFrame | None = None
        
        # Context of the LAST training run (for saving)
        self.last_run_context: Dict[str, Any] | None = None
        
        # Context of the CURRENTLY LOADED model (for prediction)
        self.loaded_model_context: Dict[str, Any] | None = None
        
        self.prediction_inputs: Dict[str, QLineEdit] = {}

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # ================= TAB 1: TRAIN =================
        train_tab = QWidget()
        self.setup_train_tab(train_tab)
        self.tabs.addTab(train_tab, "Train Pipeline")

        # ================= TAB 2: PREDICT (MODEL LIBRARY) =================
        predict_tab = QWidget()
        self.setup_predict_tab(predict_tab)
        self.tabs.addTab(predict_tab, "Model Library & Predict")

    # ================= UI SETUP HELPERS =================

    def setup_train_tab(self, tab: QWidget):
        layout = QVBoxLayout(tab)

        # File picker
        file_row = QHBoxLayout()
        self.file_label = QLabel("No CSV selected")
        browse_btn = QPushButton("Browse CSV")
        browse_btn.clicked.connect(self.on_browse_clicked)
        file_row.addWidget(self.file_label)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        # Task type
        task_row = QHBoxLayout()
        task_row.addWidget(QLabel("Task type:"))
        self.task_combo = QComboBox()
        self.task_combo.addItems(["classification", "regression", "anomaly"])
        self.task_combo.currentTextChanged.connect(self.update_model_options)
        task_row.addWidget(self.task_combo)
        layout.addLayout(task_row)

        # Algorithm
        algo_row = QHBoxLayout()
        algo_row.addWidget(QLabel("Algorithm:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["RandomForest", "LogisticRegression"])
        algo_row.addWidget(self.algo_combo)
        layout.addLayout(algo_row)

        # Target column
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target column:"))
        self.target_combo = QComboBox()
        self.target_combo.setEnabled(False)
        self.target_combo.currentIndexChanged.connect(self.on_target_changed)
        target_row.addWidget(self.target_combo)
        layout.addLayout(target_row)

        # Preprocessing
        prep_group = QGroupBox("Preprocessing Options")
        prep_layout = QVBoxLayout()
        self.scale_checkbox = QCheckBox("Scale numeric features")
        self.scale_checkbox.setChecked(True)
        prep_layout.addWidget(self.scale_checkbox)
        self.encode_checkbox = QCheckBox("One-hot encode categoricals")
        self.encode_checkbox.setChecked(True)
        prep_layout.addWidget(self.encode_checkbox)
        
        test_row = QHBoxLayout()
        test_row.addWidget(QLabel("Test size:"))
        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setRange(0.1, 0.4)
        self.test_size_spin.setSingleStep(0.05)
        self.test_size_spin.setValue(0.2)
        self.test_size_spin.setSuffix("")
        self.test_size_label = QLabel("(20%)")
        self.test_size_spin.valueChanged.connect(
            lambda v: self.test_size_label.setText(f"({int(v*100)}%)")
        )
        test_row.addWidget(self.test_size_spin)
        test_row.addWidget(self.test_size_label)
        test_row.addStretch()
        prep_layout.addLayout(test_row)
        prep_group.setLayout(prep_layout)
        layout.addWidget(prep_group)

        # Run button
        run_btn = QPushButton("Run Pipeline")
        run_btn.clicked.connect(self.on_run_clicked)
        run_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        layout.addWidget(run_btn)

        # Logs
        layout.addWidget(QLabel("Logs:"))
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(150)
        layout.addWidget(self.log_view)

        # Metrics
        layout.addWidget(QLabel("Metrics:"))
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.setMaximumHeight(100)
        layout.addWidget(self.metrics_table)

        # Save button
        self.btn_save_model = QPushButton("Save Model")
        self.btn_save_model.clicked.connect(self.on_save_model_clicked)
        self.btn_save_model.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_save_model.setEnabled(False)
        layout.addWidget(self.btn_save_model)

    def setup_predict_tab(self, tab: QWidget):
        layout = QHBoxLayout(tab)
        
        # Splitter to resize panels
        splitter = QSplitter(Qt.Horizontal)
        
        # --- LEFT PANEL: Model List ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Saved Models:"))
        
        self.model_list = QListWidget()
        self.model_list.currentItemChanged.connect(self.on_model_selected)
        left_layout.addWidget(self.model_list)
        
        refresh_btn = QPushButton("Refresh List")
        refresh_btn.clicked.connect(self.refresh_model_list)
        left_layout.addWidget(refresh_btn)
        
        splitter.addWidget(left_panel)
        
        # --- RIGHT PANEL: Details & Prediction ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Model Info Box
        self.info_box = QGroupBox("Selected Model Details")
        info_layout = QVBoxLayout()
        self.lbl_model_info = QLabel("Select a model to view details.")
        self.lbl_model_info.setWordWrap(True)
        info_layout.addWidget(self.lbl_model_info)
        self.info_box.setLayout(info_layout)
        right_layout.addWidget(self.info_box)
        
        # Input Form
        right_layout.addWidget(QLabel("Input Features:"))
        self.predict_scroll = QScrollArea()
        self.predict_scroll.setWidgetResizable(True)
        self.predict_form_widget = QWidget()
        self.predict_form_layout = QFormLayout(self.predict_form_widget)
        self.predict_scroll.setWidget(self.predict_form_widget)
        right_layout.addWidget(self.predict_scroll)
        
        # Predict Button
        self.btn_predict = QPushButton("Predict")
        self.btn_predict.clicked.connect(self.on_predict_clicked)
        self.btn_predict.setStyleSheet("font-weight: bold; padding: 10px; background-color: #e0f7fa; color: black;")
        self.btn_predict.setEnabled(False)
        right_layout.addWidget(self.btn_predict)
        
        # Result
        self.lbl_prediction_result = QLabel("Result: -")
        self.lbl_prediction_result.setStyleSheet("font-size: 18px; font-weight: bold; color: #00796b; padding: 5px;")
        self.lbl_prediction_result.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.lbl_prediction_result)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([250, 600]) # Initial ratio
        layout.addWidget(splitter)
        
        # Initial load
        self.refresh_model_list()

    # ================= LOGIC: TRAIN TAB =================

    def update_model_options(self, task: str) -> None:
        self.algo_combo.clear()
        if task == "classification":
            self.algo_combo.addItems(["RandomForest", "LogisticRegression"])
        elif task == "regression":
            self.algo_combo.addItems(["RandomForest", "LinearRegression"])
        elif task == "anomaly":
            self.algo_combo.addItems(["IsolationForest"])

    def on_browse_clicked(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Select CSV", str(Path.cwd() / "data"), "CSV Files (*.csv)"
        )
        if not path_str: return
        self.csv_path = Path(path_str)
        self.file_label.setText(self.csv_path.name)
        try:
            self.current_df = pd.read_csv(self.csv_path)
            self.target_combo.clear()
            for col in self.current_df.columns:
                self.target_combo.addItem(col)
            self.target_combo.setEnabled(True)
            self.log_view.append(f"Loaded {self.csv_path.name}: {self.current_df.shape}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def on_target_changed(self, index: int) -> None:
        if index >= 0:
            self.target_column = self.target_combo.itemText(index)

    def on_run_clicked(self) -> None:
        if not self.csv_path or not self.target_column:
            QMessageBox.warning(self, "Inputs Missing", "Select CSV and Target Column.")
            return

        self.btn_save_model.setEnabled(False)
        self.last_run_context = None

        task = self.task_combo.currentText()
        yaml_path = self._write_generated_yaml(task)

        self.log_view.append(f"Running pipeline...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            context = run_pipeline(str(yaml_path))
            self.last_run_context = context
            self.btn_save_model.setEnabled(True)
            self._show_metrics(context)
            self.log_view.append("Pipeline finished successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Pipeline Error", str(e))
            self.log_view.append(f"Error: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def _show_metrics(self, context: Dict[str, Any]) -> None:
        metrics = context.get("metrics") or context.get("anomaly_metrics") or {}
        self.metrics_table.setRowCount(0)
        for row, (k, v) in enumerate(metrics.items()):
            self.metrics_table.insertRow(row)
            self.metrics_table.setItem(row, 0, QTableWidgetItem(str(k)))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{v:.4f}"))

    def _write_generated_yaml(self, task: str) -> Path:
        import yaml
        pipeline_name = f"{task}_gui_run"
        algo = self.algo_combo.currentText()
        
        stages = [
            {"name": "load_data", "type": "csv_loader", "params": {"source": str(self.csv_path), "target_column": self.target_column}},
            {"name": "preprocess", "type": "tabular_preprocess", "params": {
                "target_column": self.target_column, "task_type": task,
                "scale_numeric": self.scale_checkbox.isChecked(),
                "encode_categoricals": self.encode_checkbox.isChecked(),
                "test_size": float(self.test_size_spin.value())
            }}
        ]
        
        model_type = "classification_rf"
        if task == "classification":
            model_type = "classification_logreg" if algo == "LogisticRegression" else "classification_rf"
        elif task == "regression":
            model_type = "regression_linear" if algo == "LinearRegression" else "regression_rf"
        elif task == "anomaly":
            model_type = "anomaly_isolation_forest"
            
        stages.append({"name": "model", "type": model_type, "params": {}})
        
        config = {"pipeline_name": pipeline_name, "stages": stages}
        yaml_path = GENERATED_DIR / f"{pipeline_name}.yml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        return yaml_path

    # ================= LOGIC: SAVE & MODEL LIBRARY =================

    def on_save_model_clicked(self) -> None:
        if not self.last_run_context: return
        
        # Generate default name
        task = self.task_combo.currentText()
        algo = self.algo_combo.currentText()
        default_name = f"{task}_{algo}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save Model", str(MODELS_DIR / default_name), "Model Files (*.pkl)"
        )
        if not path_str: return

        # Metadata Packet
        meta = {
            "task": task,
            "algorithm": algo,
            "dataset": self.csv_path.name if self.csv_path else "unknown",
            "target": self.target_column,
            "features": list(self.current_df.columns.drop(self.target_column)) if self.current_df is not None else [],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": self.last_run_context.get("metrics", {})
        }
        
        payload = {
            "meta": meta,
            "objects": {
                "model": self.last_run_context.get("model"),
                "preprocessor": self.last_run_context.get("preprocessor")
            }
        }
        
        try:
            joblib.dump(payload, path_str)
            self.log_view.append(f"Saved to {path_str}")
            self.refresh_model_list()
            QMessageBox.information(self, "Saved", "Model saved to library!")
            self.tabs.setCurrentIndex(1) # Switch to library
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def refresh_model_list(self):
        self.model_list.clear()
        if not MODELS_DIR.exists(): return
        
        files = sorted(MODELS_DIR.glob("*.pkl"), key=os.path.getmtime, reverse=True)
        for p in files:
            self.model_list.addItem(p.name)

    def on_model_selected(self, item):
        if not item: return
        filename = item.text()
        path = MODELS_DIR / filename
        
        try:
            payload = joblib.load(path)
            
            # Handle legacy format (if you saved models before this update)
            if "meta" not in payload:
                self.lbl_model_info.setText("Legacy model format (no metadata). Prediction might fail.")
                self.loaded_model_context = None
                self.btn_predict.setEnabled(False)
                return

            meta = payload["meta"]
            self.loaded_model_context = payload["objects"]
            
            # Display Info
            metrics_str = ", ".join([f"{k}={v:.3f}" for k,v in meta.get("metrics", {}).items()])
            info_text = (
                f"<b>Algorithm:</b> {meta['algorithm']} ({meta['task']})<br>"
                f"<b>Dataset:</b> {meta['dataset']} (Target: {meta['target']})<br>"
                f"<b>Date:</b> {meta['date']}<br>"
                f"<b>Metrics:</b> {metrics_str}"
            )
            self.lbl_model_info.setText(info_text)
            
            # Generate Form
            self._generate_prediction_form(meta["features"])
            self.btn_predict.setEnabled(True)
            self.lbl_prediction_result.setText("Result: -")

        except Exception as e:
            self.lbl_model_info.setText(f"Error loading model: {e}")
            self.btn_predict.setEnabled(False)

    def _generate_prediction_form(self, features: List[str]):
        # Clear old
        while self.predict_form_layout.count():
            child = self.predict_form_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        self.prediction_inputs = {}
        
        for feat in features:
            le = QLineEdit()
            le.setPlaceholderText(f"Value for {feat}")
            self.predict_form_layout.addRow(f"{feat}:", le)
            self.prediction_inputs[feat] = le

    def on_predict_clicked(self):
        if not self.loaded_model_context: return
        
        model = self.loaded_model_context["model"]
        preprocessor = self.loaded_model_context["preprocessor"]
        
        input_data = {}
        try:
            for feat, widget in self.prediction_inputs.items():
                val = widget.text().strip()
                if not val: raise ValueError(f"Missing {feat}")
                try:
                    input_data[feat] = float(val)
                except:
                    input_data[feat] = val
            
            df = pd.DataFrame([input_data])
            X = preprocessor.transform(df) if preprocessor else df
            pred = model.predict(X)[0]
            
            self.lbl_prediction_result.setText(f"Result: {pred}")
            
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", str(e))


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1100, 800) # Wider for splitter
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
