from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


_BOOL_TRUE = {"true", "1", "yes", "y", "t"}
_BOOL_FALSE = {"false", "0", "no", "n", "f"}


def _normalize_task(task: str | None) -> str:
    value = (task or "classification").strip().lower()
    if value == "binary_classification":
        return "classification"
    return value


def _coerce_series(feature: str, series: pd.Series, dtype: str) -> pd.Series:
    normalized = dtype.lower()

    if "int" in normalized:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"Column '{feature}' contains invalid integer values")
        frac = np.modf(numeric.to_numpy())[0]
        if np.any(np.abs(frac) > 1e-12):
            raise ValueError(f"Column '{feature}' contains non-integer values")
        return numeric.astype(int)

    if any(token in normalized for token in ["float", "double", "decimal"]):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"Column '{feature}' contains invalid numeric values")
        return numeric.astype(float)

    if "bool" in normalized:
        out: List[bool] = []
        for raw in series.astype(str).str.strip().str.lower().tolist():
            if raw in _BOOL_TRUE:
                out.append(True)
            elif raw in _BOOL_FALSE:
                out.append(False)
            else:
                raise ValueError(f"Column '{feature}' contains invalid boolean values")
        return pd.Series(out, index=series.index, dtype=bool)

    return series


def _normalize_output_profile(output_profile: str | None) -> str:
    profile = (output_profile or "detailed").strip().lower()
    if profile not in {"simple", "detailed"}:
        return "detailed"
    return profile


def _apply_output_profile(df: pd.DataFrame, task: str, output_profile: str) -> pd.DataFrame:
    if output_profile != "simple":
        return df

    keep = [col for col in df.columns if col == "prediction"]
    if task == "classification" and "prediction_score" in df.columns:
        keep.append("prediction_score")
    if task == "anomaly" and "anomaly_score" in df.columns:
        keep.append("anomaly_score")

    if not keep:
        keep = ["prediction"]
    return df.loc[:, keep]


def predict_dataframe(payload: Dict[str, Any], input_df: pd.DataFrame, *, output_profile: str = "detailed") -> pd.DataFrame:
    if not isinstance(payload, dict) or "meta" not in payload or "objects" not in payload:
        raise ValueError("Invalid model payload format")

    meta = payload["meta"]
    objects = payload["objects"]
    model = objects.get("model")
    preprocessor = objects.get("preprocessor")

    if model is None:
        raise ValueError("Model payload is missing model object")

    feature_columns = meta.get("feature_columns")
    feature_dtypes = meta.get("feature_dtypes", {})
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("Model payload is missing feature schema")

    missing = [col for col in feature_columns if col not in input_df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    ordered = input_df.loc[:, feature_columns].copy()
    for feature in feature_columns:
        ordered[feature] = _coerce_series(feature, ordered[feature], str(feature_dtypes.get(feature, "object")))

    X = preprocessor.transform(ordered) if preprocessor is not None else ordered
    task = _normalize_task(str(meta.get("task", "classification")))
    profile = _normalize_output_profile(output_profile)

    out = input_df.copy()

    if task == "regression":
        pred = model.predict(X)
        out["prediction"] = np.asarray(pred, dtype=float)
        return _apply_output_profile(out, task, profile)

    if task == "anomaly":
        pred = model.predict(X)
        out["prediction"] = np.where(np.asarray(pred) == -1, "Anomaly", "Normal")
        if hasattr(model, "decision_function"):
            out["anomaly_score"] = -np.asarray(model.decision_function(X), dtype=float)
        return _apply_output_profile(out, task, profile)

    pred = model.predict(X)
    out["prediction"] = pred

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        classes = list(getattr(model, "classes_", meta.get("class_labels", [])))
        if classes and probs.ndim == 2:
            for idx, class_name in enumerate(classes):
                out[f"proba_{class_name}"] = probs[:, idx]

            positive_label = meta.get("positive_label")
            if positive_label in classes:
                pos_idx = classes.index(positive_label)
                out["prediction_score"] = probs[:, pos_idx]

    return _apply_output_profile(out, task, profile)
