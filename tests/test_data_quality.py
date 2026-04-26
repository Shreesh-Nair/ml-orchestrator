from __future__ import annotations

import json

import pandas as pd

from core.data_quality import (
    analyze_data_quality,
    build_data_quality_report_payload,
    write_data_quality_report,
)


def test_analyze_data_quality_basic_summary() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "feature": [10.0, 11.0, 12.0, 13.0, 14.0],
            "target": [0, 1, 0, 1, 0],
        }
    )

    report = analyze_data_quality(df, "target")
    summary = report["summary"]

    assert summary["rows"] == 5
    assert summary["columns"] == 3
    assert summary["target"] == "target"


def test_analyze_data_quality_detects_missing_and_duplicates() -> None:
    df = pd.DataFrame(
        {
            "a": [1, 1, 2, 2, None],
            "b": ["x", "x", "y", "y", "z"],
            "target": [0, 0, 1, 1, 1],
        }
    )

    report = analyze_data_quality(df, "target")
    warnings = " | ".join(report["warnings"])

    assert "duplicate rows" in warnings
    assert "high missing values" not in warnings
    assert "a" in report["summary"]["missing_columns"]


def test_analyze_data_quality_detects_target_imbalance() -> None:
    df = pd.DataFrame(
        {
            "feature": list(range(20)),
            "target": [1] * 19 + [0],
        }
    )

    report = analyze_data_quality(df, "target")
    warnings = " | ".join(report["warnings"])

    assert "highly imbalanced" in warnings


def test_analyze_data_quality_detects_high_cardinality_and_id_columns() -> None:
    df = pd.DataFrame(
        {
            "record_id": [f"id_{i}" for i in range(50)],
            "category": [f"cat_{i}" for i in range(50)],
            "target": [0 if i < 25 else 1 for i in range(50)],
        }
    )

    report = analyze_data_quality(df, "target")
    summary = report["summary"]

    assert "record_id" in summary["potential_id_columns"]
    assert "category" in summary["high_cardinality_columns"]


def test_build_data_quality_report_payload_includes_metadata() -> None:
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )
    report = analyze_data_quality(df, "target")

    payload = build_data_quality_report_payload(
        report,
        source_csv="demo.csv",
        target_column="target",
    )

    assert payload["source_csv"] == "demo.csv"
    assert payload["target_column"] == "target"
    assert "generated_at_utc" in payload
    assert isinstance(payload["warnings"], list)


def test_write_data_quality_report_creates_json_file(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40],
            "target": [1, 1, 1, 0],
        }
    )
    report = analyze_data_quality(df, "target")
    out_path = tmp_path / "quality_report.json"

    written_path = write_data_quality_report(
        report,
        out_path,
        source_csv="sample.csv",
        target_column="target",
    )

    assert written_path == out_path
    assert out_path.exists()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["source_csv"] == "sample.csv"
    assert payload["target_column"] == "target"
    assert "summary" in payload
    assert "warnings" in payload
