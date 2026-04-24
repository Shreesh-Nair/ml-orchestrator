from pathlib import Path

import pandas as pd
import yaml
import pytest

from core.executor import run_pipeline
from core.prediction import predict_dataframe


def _run_classification_payload(tmp_path: Path) -> dict:
    data_path = tmp_path / "cls.csv"
    yaml_path = tmp_path / "pipeline.yml"

    df = pd.DataFrame(
        {
            "age": [21, 22, 23, 45, 46, 47, 31, 32, 33, 58, 59, 60],
            "city": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A", "B", "C"],
            "income": [30, 32, 31, 70, 72, 74, 45, 46, 47, 80, 82, 84],
            "label": [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        }
    )
    df.to_csv(data_path, index=False)

    config = {
        "pipeline_name": "batch_classification",
        "stages": [
            {
                "name": "load_data",
                "type": "csv_loader",
                "params": {"source": str(data_path), "target_column": "label"},
            },
            {
                "name": "preprocess",
                "type": "tabular_preprocess",
                "params": {
                    "task_type": "classification",
                    "require_binary_target": True,
                    "encode_categoricals": True,
                    "scale_numeric": True,
                    "test_size": 0.25,
                    "random_state": 42,
                },
            },
            {"name": "model", "type": "classification_rf", "params": {"random_state": 42}},
        ],
    }
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    context = run_pipeline(str(yaml_path))
    payload = {
        "meta": {
            "task": "classification",
            "feature_columns": context["feature_columns"],
            "feature_dtypes": context["feature_dtypes"],
            "class_labels": list(context.get("class_labels", [])),
            "positive_label": (context.get("artifacts") or {}).get("positive_label"),
        },
        "objects": {
            "model": context["model"],
            "preprocessor": context["preprocessor"],
        },
    }
    return payload


def test_predict_dataframe_classification_outputs_scores(tmp_path: Path) -> None:
    payload = _run_classification_payload(tmp_path)

    batch_df = pd.DataFrame(
        {
            "age": [25, 50],
            "city": ["A", "C"],
            "income": [40, 79],
        }
    )

    out = predict_dataframe(payload, batch_df)

    assert "prediction" in out.columns
    assert "prediction_score" in out.columns
    assert out.shape[0] == 2


def test_predict_dataframe_classification_simple_profile_reduces_columns(tmp_path: Path) -> None:
    payload = _run_classification_payload(tmp_path)

    batch_df = pd.DataFrame(
        {
            "age": [25, 50],
            "city": ["A", "C"],
            "income": [40, 79],
        }
    )

    out = predict_dataframe(payload, batch_df, output_profile="simple")
    assert list(out.columns) == ["prediction", "prediction_score"]


def test_predict_dataframe_missing_required_column_raises(tmp_path: Path) -> None:
    payload = _run_classification_payload(tmp_path)

    batch_df = pd.DataFrame(
        {
            "age": [25, 50],
            "income": [40, 79],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        predict_dataframe(payload, batch_df)


def test_predict_dataframe_regression_outputs_numeric_prediction() -> None:
    context = run_pipeline("examples/house_regression.yml")

    payload = {
        "meta": {
            "task": "regression",
            "feature_columns": context["feature_columns"],
            "feature_dtypes": context["feature_dtypes"],
        },
        "objects": {
            "model": context["model"],
            "preprocessor": context["preprocessor"],
        },
    }

    sample = pd.read_csv("data/housing.csv").iloc[:3]
    batch_df = sample.loc[:, context["feature_columns"]].copy()

    out = predict_dataframe(payload, batch_df)
    assert "prediction" in out.columns
    assert out["prediction"].dtype.kind in {"f", "i"}
    assert len(out) == 3

    out_simple = predict_dataframe(payload, batch_df, output_profile="simple")
    assert list(out_simple.columns) == ["prediction"]


def test_predict_dataframe_anomaly_outputs_label_and_score() -> None:
    context = run_pipeline("examples/fraud_anomaly.yml")

    payload = {
        "meta": {
            "task": "anomaly",
            "feature_columns": context["feature_columns"],
            "feature_dtypes": context["feature_dtypes"],
        },
        "objects": {
            "model": context["anomaly_model"],
            "preprocessor": context["preprocessor"],
        },
    }

    sample = pd.read_csv("data/fraud.csv").iloc[:5]
    batch_df = sample.loc[:, context["feature_columns"]].copy()

    out = predict_dataframe(payload, batch_df)
    assert "prediction" in out.columns
    assert "anomaly_score" in out.columns
    assert set(out["prediction"].unique()).issubset({"Normal", "Anomaly"})

    out_simple = predict_dataframe(payload, batch_df, output_profile="simple")
    assert list(out_simple.columns) == ["prediction", "anomaly_score"]
