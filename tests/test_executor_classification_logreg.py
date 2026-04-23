from pathlib import Path

import pandas as pd
import yaml

from core.executor import run_pipeline


def test_run_logistic_classification_pipeline_has_metrics(tmp_path: Path) -> None:
    data_path = tmp_path / "cls.csv"
    yaml_path = tmp_path / "pipeline.yml"

    df = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "x2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "city": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
            "label": [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        }
    )
    df.to_csv(data_path, index=False)

    config = {
        "pipeline_name": "classification_logreg_test",
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
                    "scale_numeric": True,
                    "encode_categoricals": True,
                    "test_size": 0.2,
                    "random_state": 42,
                },
            },
            {
                "name": "model",
                "type": "classification_logreg",
                "params": {"max_iter": 200, "C": 1.0, "random_state": 42},
            },
        ],
    }

    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    context = run_pipeline(str(yaml_path))

    assert "model" in context
    assert "metrics" in context
    metrics = context["metrics"]
    assert "accuracy" in metrics
    assert "f1" in metrics

    artifacts = context["artifacts"]
    assert "y_test" in artifacts
    assert "y_pred" in artifacts
    assert "y_proba" in artifacts
