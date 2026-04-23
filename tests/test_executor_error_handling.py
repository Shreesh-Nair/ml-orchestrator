from pathlib import Path

import pandas as pd
import pytest
import yaml

from core.executor import PipelineExecutionError, run_pipeline


def _write_pipeline(tmp_path: Path, source: str, *, target: str = "label", task: str = "classification") -> Path:
    yaml_path = tmp_path / "pipeline.yml"
    config = {
        "pipeline_name": "error_case",
        "stages": [
            {
                "name": "load_data",
                "type": "csv_loader",
                "params": {"source": source, "target_column": target},
            },
            {
                "name": "preprocess",
                "type": "tabular_preprocess",
                "params": {
                    "task_type": task,
                    "require_binary_target": task in {"classification", "anomaly"},
                    "scale_numeric": True,
                    "encode_categoricals": True,
                    "test_size": 0.25,
                    "random_state": 42,
                },
            },
            {"name": "model", "type": "classification_rf", "params": {"random_state": 42}},
        ],
    }
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return yaml_path


def test_pipeline_wraps_missing_csv_with_hint(tmp_path: Path) -> None:
    yaml_path = _write_pipeline(tmp_path, "does_not_exist.csv")

    with pytest.raises(PipelineExecutionError) as exc_info:
        run_pipeline(str(yaml_path))

    message = str(exc_info.value)
    assert "Stage 'load_data'" in message
    assert "csv_loader" in message
    assert "Check the dataset path" in message


def test_pipeline_wraps_missing_target_with_hint(tmp_path: Path) -> None:
    data_path = tmp_path / "dataset.csv"
    pd.DataFrame({"x1": [1, 2, 3, 4], "x2": [5, 6, 7, 8]}).to_csv(data_path, index=False)

    yaml_path = _write_pipeline(tmp_path, str(data_path), target="missing_target")

    with pytest.raises(PipelineExecutionError) as exc_info:
        run_pipeline(str(yaml_path))

    message = str(exc_info.value)
    assert "Stage 'load_data'" in message
    assert "target column 'missing_target' is missing" in message
    assert "Verify the selected target column" in message


def test_pipeline_preprocess_requires_feature_columns(tmp_path: Path) -> None:
    data_path = tmp_path / "dataset.csv"
    pd.DataFrame({"label": [0, 1, 0, 1]}).to_csv(data_path, index=False)

    yaml_path = _write_pipeline(tmp_path, str(data_path), target="label")

    with pytest.raises(PipelineExecutionError) as exc_info:
        run_pipeline(str(yaml_path))

    message = str(exc_info.value)
    assert "Stage 'preprocess'" in message
    assert "at least one feature column" in message


def test_pipeline_preprocess_requires_minimum_rows(tmp_path: Path) -> None:
    data_path = tmp_path / "dataset.csv"
    pd.DataFrame({"x1": [1, 2, 3], "label": [0, 1, 0]}).to_csv(data_path, index=False)

    yaml_path = _write_pipeline(tmp_path, str(data_path), target="label")

    with pytest.raises(PipelineExecutionError) as exc_info:
        run_pipeline(str(yaml_path))

    message = str(exc_info.value)
    assert "Stage 'preprocess'" in message
    assert "at least 4 rows" in message
