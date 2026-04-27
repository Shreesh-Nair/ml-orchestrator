from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def apply_quick_fixes(
    df: pd.DataFrame,
    *,
    target_column: str | None = None,
    drop_constant_columns: bool = True,
    missing_strategy: str = "none",
) -> Tuple[pd.DataFrame, List[str]]:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("apply_quick_fixes expects a pandas DataFrame")

    fixed = df.copy(deep=True)
    actions: List[str] = []

    if drop_constant_columns:
        constant_cols: List[str] = []
        for col in fixed.columns:
            if target_column and col == target_column:
                continue
            if fixed[col].nunique(dropna=False) <= 1:
                constant_cols.append(col)

        if constant_cols:
            fixed = fixed.drop(columns=constant_cols)
            actions.append(f"Dropped constant columns: {constant_cols}")

    if target_column and target_column in fixed.columns:
        missing_target_rows = int(fixed[target_column].isna().sum())
        if missing_target_rows > 0:
            fixed = fixed[fixed[target_column].notna()].copy()
            actions.append(
                f"Dropped {missing_target_rows} rows with missing target '{target_column}'."
            )

    strategy = (missing_strategy or "none").strip().lower()
    valid_strategies = {"none", "drop_rows", "fill_simple"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Invalid missing strategy '{missing_strategy}'. "
            f"Expected one of: {sorted(valid_strategies)}"
        )

    if strategy == "drop_rows":
        before = len(fixed)
        fixed = fixed.dropna().copy()
        dropped = before - len(fixed)
        if dropped > 0:
            actions.append(f"Dropped {dropped} rows containing missing values.")

    if strategy == "fill_simple":
        fill_counts: Dict[str, int] = {}
        for col in fixed.columns:
            if target_column and col == target_column:
                continue

            missing_count = int(fixed[col].isna().sum())
            if missing_count <= 0:
                continue

            if pd.api.types.is_numeric_dtype(fixed[col]):
                median_val = fixed[col].median()
                if pd.isna(median_val):
                    median_val = 0
                fixed[col] = fixed[col].fillna(median_val)
            else:
                modes = fixed[col].mode(dropna=True)
                fill_val = modes.iloc[0] if len(modes) > 0 else "missing"
                fixed[col] = fixed[col].fillna(fill_val)

            fill_counts[col] = missing_count

        if fill_counts:
            actions.append(f"Filled missing values in columns: {fill_counts}")

    fixed = fixed.reset_index(drop=True)
    return fixed, actions
