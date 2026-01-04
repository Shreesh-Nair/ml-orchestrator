import sys
import os
import datetime
from pathlib import Path
from typing import Dict, Any, List
import joblib

# Matplotlib for PySide6
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

# Make project root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QFileDialog, QTableWidget,
    QTableWidgetItem, QMessageBox, QCheckBox, QDoubleSpinBox, QTabWidget,
    QScrollArea, QFormLayout, QLineEdit, QSplitter, QListWidget, QGroupBox
)
from PySide6.QtCore import Qt
from core.executor import run_pipeline

EXAMPLES_DIR = Path("examples")
GENERATED_DIR = EXAMPLES_DIR / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- Matplotlib Canvas Class ---
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Increased default figsize slightly
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ML Orchestrator GUI")
        self.csv_path: Path | None = None
        self.target_column: str | None = None
        self.current_df: pd.DataFrame | None = None
        
        self.last_run_context: Dict[str, Any] | None = None
        self.loaded_model_context: Dict[str, Any] | None = None
        self.prediction_inputs: Dict[str, QLineEdit] = {}

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab 1: Train
        train_tab = QWidget()
        self.setup_train_tab(train_tab)
        self.tabs.addTab(train_tab, "Train Pipeline")

        # Tab 2: Predict
        predict_tab = QWidget()
        self.setup_predict_tab(predict_tab)
        self.tabs.addTab(predict_tab, "Model Library & Predict")

    def setup_train_tab(self, tab: QWidget):
        splitter = QSplitter(Qt.Vertical)
        layout = QVBoxLayout(tab)
        layout.addWidget(splitter)

        # --- Top Section: Controls ---
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)

        # File Picker
        file_row = QHBoxLayout()
        self.file_label = QLabel("No CSV selected")
        browse_btn = QPushButton("Browse CSV")
        browse_btn.clicked.connect(self.on_browse_clicked)
        file_row.addWidget(self.file_label)
        file_row.addWidget(browse_btn)
        top_layout.addLayout(file_row)

        # Config Rows
        task_row = QHBoxLayout()
        task_row.addWidget(QLabel("Task type:"))
        self.task_combo = QComboBox()
        self.task_combo.addItems(["classification", "regression", "anomaly"])
        self.task_combo.currentTextChanged.connect(self.update_model_options)
        task_row.addWidget(self.task_combo)
        top_layout.addLayout(task_row)

        algo_row = QHBoxLayout()
        algo_row.addWidget(QLabel("Algorithm:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["RandomForest", "LogisticRegression"])
        algo_row.addWidget(self.algo_combo)
        top_layout.addLayout(algo_row)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target column:"))
        self.target_combo = QComboBox()
        self.target_combo.setEnabled(False)
        self.target_combo.currentIndexChanged.connect(self.on_target_changed)
        target_row.addWidget(self.target_combo)
        top_layout.addLayout(target_row)

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
        top_layout.addWidget(prep_group)

        # Run Button
        run_btn = QPushButton("Run Pipeline")
        run_btn.clicked.connect(self.on_run_clicked)
        run_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        top_layout.addWidget(run_btn)

        # Metrics Table (Small)
        top_layout.addWidget(QLabel("Metrics:"))
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.setMaximumHeight(80)
        top_layout.addWidget(self.metrics_table)
        
        # Save Button
        self.btn_save_model = QPushButton("Save Model")
        self.btn_save_model.clicked.connect(self.on_save_model_clicked)
        self.btn_save_model.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_save_model.setEnabled(False)
        top_layout.addWidget(self.btn_save_model)

        splitter.addWidget(top_widget)

        # --- Bottom Section: Visualizations ---
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        viz_layout.addWidget(QLabel("Visualizations:"))
        
        self.viz_tabs = QTabWidget()
        self.viz_tabs.addTab(QLabel("Run a pipeline to see plots."), "Info")
        viz_layout.addWidget(self.viz_tabs)
        
        splitter.addWidget(viz_widget)
        splitter.setSizes([450, 450]) 

    def setup_predict_tab(self, tab: QWidget):
        layout = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Horizontal)
        
        # Left Panel
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
        
        # Right Panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.info_box = QGroupBox("Selected Model Details")
        info_layout = QVBoxLayout()
        self.lbl_model_info = QLabel("Select a model to view details.")
        self.lbl_model_info.setWordWrap(True)
        info_layout.addWidget(self.lbl_model_info)
        self.info_box.setLayout(info_layout)
        right_layout.addWidget(self.info_box)
        
        right_layout.addWidget(QLabel("Input Features:"))
        self.predict_scroll = QScrollArea()
        self.predict_scroll.setWidgetResizable(True)
        self.predict_form_widget = QWidget()
        self.predict_form_layout = QFormLayout(self.predict_form_widget)
        self.predict_scroll.setWidget(self.predict_form_widget)
        right_layout.addWidget(self.predict_scroll)
        
        self.btn_predict = QPushButton("Predict")
        self.btn_predict.clicked.connect(self.on_predict_clicked)
        self.btn_predict.setStyleSheet("font-weight: bold; padding: 10px; background-color: #e0f7fa; color: black;")
        self.btn_predict.setEnabled(False)
        right_layout.addWidget(self.btn_predict)
        
        self.lbl_prediction_result = QLabel("Result: -")
        self.lbl_prediction_result.setStyleSheet("font-size: 18px; font-weight: bold; color: #00796b; padding: 5px;")
        self.lbl_prediction_result.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.lbl_prediction_result)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([250, 600])
        layout.addWidget(splitter)
        self.refresh_model_list()

    def update_model_options(self, task: str) -> None:
        self.algo_combo.clear()
        if task == "classification":
            self.algo_combo.addItems(["RandomForest", "LogisticRegression"])
        elif task == "regression":
            self.algo_combo.addItems(["RandomForest", "LinearRegression"])
        elif task == "anomaly":
            self.algo_combo.addItems(["IsolationForest"])

    def on_browse_clicked(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Select CSV", str(Path.cwd() / "data"), "CSV Files (*.csv)")
        if not path_str: return
        self.csv_path = Path(path_str)
        self.file_label.setText(self.csv_path.name)
        try:
            self.current_df = pd.read_csv(self.csv_path)
            self.target_combo.clear()
            for col in self.current_df.columns:
                self.target_combo.addItem(col)
            self.target_combo.setEnabled(True)
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
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            context = run_pipeline(str(yaml_path))
            self.last_run_context = context
            self.btn_save_model.setEnabled(True)
            self._show_metrics(context)
            self._update_visualizations(context)
        except Exception as e:
            QMessageBox.critical(self, "Pipeline Error", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    def _show_metrics(self, context: Dict[str, Any]) -> None:
        metrics = context.get("metrics") or context.get("anomaly_metrics") or {}
        self.metrics_table.setRowCount(0)
        for row, (k, v) in enumerate(metrics.items()):
            self.metrics_table.insertRow(row)
            self.metrics_table.setItem(row, 0, QTableWidgetItem(str(k)))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{v:.4f}"))

    # ================= VISUALIZATION FIXES =================
    def _update_visualizations(self, context: Dict[str, Any]):
        """Parses 'artifacts' from context and plots them."""
        self.viz_tabs.clear()
        artifacts = context.get("artifacts", {})
        
        if not artifacts:
            self.viz_tabs.addTab(QLabel("No visualization data available for this model."), "Info")
            return

        # 1. Confusion Matrix
        if "y_test" in artifacts and "y_pred" in artifacts:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            y_test = artifacts["y_test"]
            y_pred = artifacts["y_pred"]
            
            # Compute matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot
            canvas = MplCanvas(self, width=5, height=4, dpi=100)
            ax = canvas.axes
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            
            # Plot with smaller font for values inside cells if needed, but default is usually okay
            disp.plot(ax=ax, cmap="Blues", colorbar=True)
            ax.set_title("Confusion Matrix", fontsize=10)
            
            # FIX: Tight Layout to prevent clipping
            canvas.figure.tight_layout()
            self.viz_tabs.addTab(canvas, "Confusion Matrix")

        # 2. Feature Importance
        if "feature_importance" in artifacts:
            importances = artifacts["feature_importance"]
            names = artifacts.get("feature_names", [f"F{i}" for i in range(len(importances))])
            
            # Sort indices
            indices = np.argsort(importances)[::-1]
            top_n = 10 # Limit to top 10
            indices = indices[:top_n]
            
            canvas = MplCanvas(self, width=5, height=4, dpi=100)
            ax = canvas.axes
            
            # Bar plot
            ax.bar(range(len(indices)), importances[indices], align="center")
            ax.set_xticks(range(len(indices)))
            
            # Clean up long names: split by '__' and take last part, truncate if >15 chars
            clean_names = []
            for i in indices:
                s = str(names[i]).split("__")[-1]
                if len(s) > 15:
                    s = s[:12] + "..."
                clean_names.append(s)
                
            ax.set_xticklabels(clean_names, rotation=45, ha="right", fontsize=9)
            ax.set_title("Top 10 Feature Importances", fontsize=10)
            
            # FIX: Explicit bottom margin for rotated labels
            canvas.figure.subplots_adjust(bottom=0.25) 
            # tight_layout might override subplots_adjust, usually safe to call tight_layout afterwards 
            # but sometimes tight_layout fails with severe rotation. 
            # Using tight_layout with padding usually works best.
            canvas.figure.tight_layout()
            
            canvas.draw()
            self.viz_tabs.addTab(canvas, "Feature Importance")

        # 3. ROC Curve
        if "y_proba" in artifacts and "y_test" in artifacts:
            y_proba = artifacts["y_proba"]
            y_test = artifacts["y_test"]
            if y_proba.shape[1] == 2: # Binary
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                canvas = MplCanvas(self, width=5, height=4, dpi=100)
                ax = canvas.axes
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate', fontsize=9)
                ax.set_ylabel('True Positive Rate', fontsize=9)
                ax.set_title('ROC Curve', fontsize=10)
                ax.legend(loc="lower right", fontsize=9)
                
                # FIX: Tight layout
                canvas.figure.tight_layout()
                self.viz_tabs.addTab(canvas, "ROC Curve")

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
        task = self.task_combo.currentText()
        algo = self.algo_combo.currentText()
        default_name = f"{task}_{algo}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        path_str, _ = QFileDialog.getSaveFileName(self, "Save Model", str(MODELS_DIR / default_name), "Model Files (*.pkl)")
        if not path_str: return
        
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
            self.refresh_model_list()
            QMessageBox.information(self, "Saved", "Model saved to library!")
            self.tabs.setCurrentIndex(1)
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
            if "meta" not in payload:
                self.lbl_model_info.setText("Legacy model format (no metadata). Prediction might fail.")
                self.loaded_model_context = None
                self.btn_predict.setEnabled(False)
                return
            
            meta = payload["meta"]
            self.loaded_model_context = payload["objects"]
            metrics_str = ", ".join([f"{k}={v:.3f}" for k,v in meta.get("metrics", {}).items()])
            info_text = (f"Algorithm: {meta['algorithm']} ({meta['task']})\n"
                         f"Dataset: {meta['dataset']} (Target: {meta['target']})\n"
                         f"Date: {meta['date']}\n"
                         f"Metrics: {metrics_str}")
            self.lbl_model_info.setText(info_text)
            self._generate_prediction_form(meta["features"])
            self.btn_predict.setEnabled(True)
            self.lbl_prediction_result.setText("Result: -")
        except Exception as e:
            self.lbl_model_info.setText(f"Error loading model: {e}")
            self.btn_predict.setEnabled(False)

    def _generate_prediction_form(self, features: List[str]):
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
    win.resize(1100, 850)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
