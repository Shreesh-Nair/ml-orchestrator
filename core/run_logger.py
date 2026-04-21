# core/run_logger.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import csv


LOGS_DIR = Path("logs")
RUNS_CSV = LOGS_DIR / "runs.csv"


def log_run(pipeline_name: str, context: Dict[str, Any]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = context.get("metrics") or {}
    anomaly_metrics = context.get("anomaly_metrics") or {}

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_name": pipeline_name,
        "accuracy": metrics.get("accuracy"),
        "classification_precision": metrics.get("precision"),
        "classification_recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "anomaly_auc": anomaly_metrics.get("auc"),
        "anomaly_precision": anomaly_metrics.get("precision"),
        "anomaly_recall": anomaly_metrics.get("recall"),
        "anomaly_f1": anomaly_metrics.get("f1"),
    }

    file_exists = RUNS_CSV.exists()

    with RUNS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
