from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import yaml

from core.executor import run_pipeline
from core.run_logger import CANONICAL_FIELDS, log_run


def test_log_run_writes_canonical_header_on_new_file(tmp_path: Path, monkeypatch) -> None:
    runs_csv = tmp_path / "runs.csv"
    monkeypatch.setattr("core.run_logger.get_runs_csv_path", lambda: runs_csv)

    log_run(
        "classification_demo",
        {
            "metrics": {
                "accuracy": 0.81,
                "precision": 0.75,
                "recall": 0.65,
                "f1": 0.69,
                "roc_auc": 0.83,
            }
        },
    )

    with runs_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == CANONICAL_FIELDS
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["pipeline_name"] == "classification_demo"
    assert rows[0]["classification_precision"] == "0.75"
    assert rows[0]["roc_auc"] == "0.83"


def test_log_run_appends_to_legacy_header(tmp_path: Path, monkeypatch) -> None:
    runs_csv = tmp_path / "runs.csv"
    legacy_header = ["timestamp", "pipeline_name", "accuracy", "f1", "auc", "precision", "recall", "anomaly_f1"]
    runs_csv.write_text(",".join(legacy_header) + "\n", encoding="utf-8")

    monkeypatch.setattr("core.run_logger.get_runs_csv_path", lambda: runs_csv)

    log_run(
        "anomaly_demo",
        {"anomaly_metrics": {"auc": 0.95, "precision": 0.10, "recall": 0.66, "f1": 0.18}},
    )

    with runs_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == legacy_header
    assert len(rows) == 1
    assert rows[0]["pipeline_name"] == "anomaly_demo"
    assert rows[0]["auc"] == "0.95"
    assert rows[0]["precision"] == "0.1"
    assert rows[0]["recall"] == "0.66"
    assert rows[0]["anomaly_f1"] == "0.18"


def test_run_pipeline_resolves_csv_relative_to_yaml(tmp_path: Path) -> None:
    data_path = tmp_path / "dataset.csv"
    yaml_path = tmp_path / "pipeline.yml"

    df = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6, 7, 8],
            "x2": ["A", "A", "A", "B", "B", "B", "A", "B"],
            "label": [0, 0, 0, 1, 1, 1, 0, 1],
        }
    )
    df.to_csv(data_path, index=False)

    config = {
        "pipeline_name": "relative_path_test",
        "stages": [
            {
                "name": "load_data",
                "type": "csv_loader",
                "params": {"source": "dataset.csv", "target_column": "label"},
            },
            {
                "name": "preprocess",
                "type": "tabular_preprocess",
                "params": {
                    "task_type": "classification",
                    "require_binary_target": True,
                    "scale_numeric": True,
                    "encode_categoricals": True,
                    "test_size": 0.25,
                },
            },
            {"name": "model", "type": "classification_rf", "params": {}},
        ],
    }
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    context = run_pipeline(str(yaml_path))
    assert "model" in context
    assert "metrics" in context
