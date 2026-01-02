import sys
from pathlib import Path
from typing import Dict, Any

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
)
from PySide6.QtCore import Qt

from core.executor import run_pipeline

EXAMPLES_DIR = Path("examples")
GENERATED_DIR = EXAMPLES_DIR / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ML Orchestrator GUI")

        self.csv_path: Path | None = None
        self.target_column: str | None = None
        self.current_df: pd.DataFrame | None = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Row: file picker
        file_row = QHBoxLayout()
        self.file_label = QLabel("No CSV selected")
        browse_btn = QPushButton("Browse CSV")
        browse_btn.clicked.connect(self.on_browse_clicked)
        file_row.addWidget(self.file_label)
        file_row.addWidget(browse_btn)
        main_layout.addLayout(file_row)

        # Row: task type
        task_row = QHBoxLayout()
        task_row.addWidget(QLabel("Task type:"))
        self.task_combo = QComboBox()
        self.task_combo.addItems(["classification", "regression", "anomaly"])
        self.task_combo.currentTextChanged.connect(self.update_model_options)  # ✅ Connect update
        task_row.addWidget(self.task_combo)
        main_layout.addLayout(task_row)

        # 🔥 NEW: Algorithm Dropdown
        algo_row = QHBoxLayout()
        algo_row.addWidget(QLabel("Algorithm:"))
        self.algo_combo = QComboBox()
        # Initialize with classification models (default task)
        self.algo_combo.addItems(["RandomForest", "LogisticRegression"])
        algo_row.addWidget(self.algo_combo)
        main_layout.addLayout(algo_row)

        # Row: target column
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target column:"))
        self.target_combo = QComboBox()
        self.target_combo.setEnabled(False)
        self.target_combo.currentIndexChanged.connect(self.on_target_changed)
        target_row.addWidget(self.target_combo)
        main_layout.addLayout(target_row)

        # Preprocessing controls
        prep_label = QLabel("Preprocessing options:")
        prep_label.setStyleSheet("font-weight: bold; padding: 5px;")
        main_layout.addWidget(prep_label)
        
        self.scale_checkbox = QCheckBox("Scale numeric features")
        self.scale_checkbox.setChecked(True)
        main_layout.addWidget(self.scale_checkbox)
        
        self.encode_checkbox = QCheckBox("One-hot encode categoricals")
        self.encode_checkbox.setChecked(True)
        main_layout.addWidget(self.encode_checkbox)
        
        test_row = QHBoxLayout()
        test_row.addWidget(QLabel("Test size:"))
        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setRange(0.1, 0.4)
        self.test_size_spin.setSingleStep(0.05)
        self.test_size_spin.setValue(0.2)
        self.test_size_spin.setSuffix(" (")
        self.test_size_spin.setDecimals(2)
        self.test_size_label = QLabel("20%)")
        test_row.addWidget(self.test_size_spin)
        test_row.addWidget(self.test_size_label)
        test_row.addStretch()
        main_layout.addLayout(test_row)

        self.test_size_spin.valueChanged.connect(
            lambda v: self.test_size_label.setText(f"{int(v*100)}%)")
        )

        # Run button
        run_btn = QPushButton("Run pipeline")
        run_btn.clicked.connect(self.on_run_clicked)
        run_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        main_layout.addWidget(run_btn)

        # Logs
        main_layout.addWidget(QLabel("Logs:"))
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        main_layout.addWidget(self.log_view)

        # Metrics table
        main_layout.addWidget(QLabel("Metrics:"))
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        main_layout.addWidget(self.metrics_table)

    # ---------- UI callbacks ----------

    def update_model_options(self, task: str) -> None:
        """Update the algorithm dropdown based on selected task."""
        self.algo_combo.clear()
        if task == "classification":
            self.algo_combo.addItems(["RandomForest", "LogisticRegression"])
        elif task == "regression":
            self.algo_combo.addItems(["RandomForest", "LinearRegression"])
        elif task == "anomaly":
            self.algo_combo.addItems(["IsolationForest"])

    def on_browse_clicked(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV file",
            str(Path.cwd() / "data"),
            "CSV Files (*.csv)",
        )
        if not path_str:
            return

        self.csv_path = Path(path_str)
        self.file_label.setText(str(self.csv_path))

        try:
            self.current_df = pd.read_csv(self.csv_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error", f"Failed to read CSV:\n{exc}")
            self.current_df = None
            self.target_combo.clear()
            self.target_combo.setEnabled(False)
            return

        self.target_combo.clear()
        for col in self.current_df.columns:
            self.target_combo.addItem(col)
        self.target_combo.setEnabled(True)
        self.log_view.append(f"Loaded CSV with shape {self.current_df.shape}")

    def on_target_changed(self, index: int) -> None:
        if index >= 0:
            self.target_column = self.target_combo.itemText(index)

    def on_run_clicked(self) -> None:
        if self.csv_path is None or self.target_column is None:
            QMessageBox.warning(
                self,
                "Missing inputs",
                "Please select a CSV and target column first.",
            )
            return

        task = self.task_combo.currentText()
        yaml_path = self._write_generated_yaml(task)

        self.log_view.append(f"Running pipeline: {yaml_path}")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            context = run_pipeline(str(yaml_path))
        except Exception as exc:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error while running pipeline", str(exc))
            self.log_view.append(f"Error: {exc}")
            return

        QApplication.restoreOverrideCursor()
        self._show_metrics(context)
        self.log_view.append("Pipeline finished.")

    # ---------- Helpers ----------

    def _write_generated_yaml(self, task: str) -> Path:
        """
        Generate a YAML file under examples/generated/ using:
        - task type (classification/regression)
        - selected algorithm (RF/Linear/Logistic)
        - preprocessing options
        """
        import yaml  # Local import

        pipeline_name = f"{task}_gui_run"
        algo = self.algo_combo.currentText()  # e.g., "LogisticRegression"

        stages = [
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
                    "task_type": task,
                    "scale_numeric": self.scale_checkbox.isChecked(),
                    "encode_categoricals": self.encode_checkbox.isChecked(),
                    "test_size": float(self.test_size_spin.value()),
                },
            },
        ]

        # ✅ Map GUI selection to handler_registry keys
        model_type = "classification_rf"  # Default fallback

        if task == "classification":
            if algo == "LogisticRegression":
                model_type = "classification_logreg"
            else:
                model_type = "classification_rf"
        
        elif task == "regression":
            if algo == "LinearRegression":
                model_type = "regression_linear"
            else:
                model_type = "regression_rf"
        
        elif task == "anomaly":
            model_type = "anomaly_isolation_forest"

        stages.append({
            "name": "model",
            "type": model_type,
            "params": {}
        })

        config = {
            "pipeline_name": pipeline_name,
            "stages": stages,
        }

        yaml_path = GENERATED_DIR / f"{pipeline_name}.yml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        return yaml_path

    def _show_metrics(self, context: Dict[str, Any]) -> None:
        metrics = context.get("metrics") or context.get("anomaly_metrics") or {}
        self.metrics_table.setRowCount(0)
        for row, (key, value) in enumerate(metrics.items()):
            self.metrics_table.insertRow(row)
            self.metrics_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{value:.4f}"))


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(900, 750)  # Taller for extra controls
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
