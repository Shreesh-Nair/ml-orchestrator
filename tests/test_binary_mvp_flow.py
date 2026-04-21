from pathlib import Path

import pandas as pd
import yaml

from core.executor import run_pipeline


def test_binary_classification_mvp_end_to_end(tmp_path: Path) -> None:
    data_path = tmp_path / "binary_data.csv"
    yaml_path = tmp_path / "pipeline.yml"

    # Simple binary dataset with mixed numeric/categorical features.
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
        "pipeline_name": "binary_mvp_test",
        "stages": [
            {
                "name": "load_data",
                "type": "csv_loader",
                "params": {
                    "source": str(data_path),
                    "target_column": "label",
                },
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
                },
            },
            {
                "name": "model",
                "type": "classification_rf",
                "params": {},
            },
        ],
    }

    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    context = run_pipeline(str(yaml_path))

    assert "model" in context
    assert "preprocessor" in context
    assert "metrics" in context
    assert "artifacts" in context

    metrics = context["metrics"]
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in metrics

    artifacts = context["artifacts"]
    assert "y_test" in artifacts
    assert "y_pred" in artifacts
    assert "y_proba" in artifacts
    assert "classes" in artifacts
    assert "positive_label" in artifacts
