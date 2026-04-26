from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def analyze_data_quality(df: pd.DataFrame, target_column: str | None = None) -> Dict[str, Any]:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("analyze_data_quality expects a pandas DataFrame")

    row_count, column_count = df.shape
    warnings: List[str] = []

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        warnings.append(f"{duplicate_rows} duplicate rows detected.")

    missing_columns: Dict[str, float] = {}
    for col in df.columns:
        pct = float(df[col].isna().mean())
        if pct > 0:
            missing_columns[col] = pct
            if pct >= 0.30:
                warnings.append(f"Column '{col}' has high missing values ({pct * 100:.1f}%).")

    constant_columns: List[str] = []
    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            constant_columns.append(col)
    if constant_columns:
        warnings.append(f"Constant columns found: {constant_columns}")

    high_cardinality_columns: List[str] = []
    potential_id_columns: List[str] = []
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue

        unique_count = int(series.nunique())
        unique_ratio = _safe_ratio(unique_count, len(series))

        if series.dtype == "object" and len(series) >= 20 and unique_ratio >= 0.80:
            high_cardinality_columns.append(col)

        if len(series) >= 20 and unique_ratio >= 0.98:
            potential_id_columns.append(col)

    if high_cardinality_columns:
        warnings.append(
            "High-cardinality categorical columns may reduce generalization: "
            f"{high_cardinality_columns}"
        )

    leakage_name_hints: List[str] = []
    if target_column and target_column in df.columns:
        normalized_target = target_column.strip().lower()
        for col in df.columns:
            if col == target_column:
                continue
            name = col.strip().lower()
            if normalized_target and (normalized_target in name or name in normalized_target):
                leakage_name_hints.append(col)

        target_series = df[target_column].dropna()
        if not target_series.empty and target_series.nunique() <= 10:
            distribution = target_series.value_counts(normalize=True)
            max_share = float(distribution.max())
            if max_share >= 0.90:
                warnings.append(
                    f"Target '{target_column}' is highly imbalanced (majority class {max_share * 100:.1f}%)."
                )

    if leakage_name_hints:
        warnings.append(
            "Potential leakage by column name similarity to target: "
            f"{leakage_name_hints}"
        )

    outlier_columns: List[str] = []
    numeric_df = df.select_dtypes(include=["number"])
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) < 10:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_ratio = float(((series < lower) | (series > upper)).mean())
        if outlier_ratio >= 0.20:
            outlier_columns.append(col)

    if outlier_columns:
        warnings.append(f"Potential heavy outliers in numeric columns: {outlier_columns}")

    summary = {
        "rows": row_count,
        "columns": column_count,
        "duplicate_rows": duplicate_rows,
        "missing_columns": missing_columns,
        "constant_columns": constant_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "potential_id_columns": potential_id_columns,
        "outlier_columns": outlier_columns,
        "target": target_column,
    }

    return {
        "summary": summary,
        "warnings": warnings,
    }


def build_data_quality_report_payload(
    report: Dict[str, Any],
    *,
    source_csv: str | None = None,
    target_column: str | None = None,
) -> Dict[str, Any]:
    if not isinstance(report, dict):
        raise ValueError("report must be a dictionary")

    summary = report.get("summary")
    warnings = report.get("warnings")
    if not isinstance(summary, dict) or not isinstance(warnings, list):
        raise ValueError("report must include 'summary' (dict) and 'warnings' (list)")

    return {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_csv": source_csv,
        "target_column": target_column,
        "summary": summary,
        "warnings": warnings,
    }


def write_data_quality_report(
    report: Dict[str, Any],
    output_path: Path,
    *,
    source_csv: str | None = None,
    target_column: str | None = None,
) -> Path:
    payload = build_data_quality_report_payload(
        report,
        source_csv=source_csv,
        target_column=target_column,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
