# core/run_logger.py
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from core.paths import get_runs_csv_path


CANONICAL_FIELDS: List[str] = [
    "timestamp",
    "pipeline_name",
    "accuracy",
    "classification_precision",
    "classification_recall",
    "f1",
    "roc_auc",
    "anomaly_auc",
    "anomaly_precision",
    "anomaly_recall",
    "anomaly_f1",
]


def _read_header(path: Path) -> List[str] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if row:
                return row
    return None


def _full_row(pipeline_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
    metrics = context.get("metrics") or {}
    anomaly_metrics = context.get("anomaly_metrics") or {}

    return {
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


def _legacy_row(full_row: Dict[str, Any]) -> Dict[str, Any]:
    roc_auc = full_row.get("roc_auc")
    anomaly_auc = full_row.get("anomaly_auc")
    cls_precision = full_row.get("classification_precision")
    cls_recall = full_row.get("classification_recall")
    an_precision = full_row.get("anomaly_precision")
    an_recall = full_row.get("anomaly_recall")

    return {
        "timestamp": full_row.get("timestamp"),
        "pipeline_name": full_row.get("pipeline_name"),
        "accuracy": full_row.get("accuracy"),
        "f1": full_row.get("f1"),
        "auc": roc_auc if roc_auc is not None else anomaly_auc,
        "precision": cls_precision if cls_precision is not None else an_precision,
        "recall": cls_recall if cls_recall is not None else an_recall,
        "anomaly_f1": full_row.get("anomaly_f1"),
    }


def _row_for_header(header: List[str], full_row: Dict[str, Any]) -> Dict[str, Any]:
    legacy = _legacy_row(full_row)
    row: Dict[str, Any] = {}

    for field in header:
        if field in full_row:
            row[field] = full_row[field]
        elif field in legacy:
            row[field] = legacy[field]
        else:
            row[field] = None

    return row


def log_run(pipeline_name: str, context: Dict[str, Any]) -> None:
    runs_csv = get_runs_csv_path()
    runs_csv.parent.mkdir(parents=True, exist_ok=True)

    full = _full_row(pipeline_name, context)
    existing_header = _read_header(runs_csv)

    header = existing_header if existing_header else CANONICAL_FIELDS
    row = _row_for_header(header, full)

    with runs_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        if existing_header is None:
            writer.writeheader()
        writer.writerow(row)
