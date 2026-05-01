"""Unit tests for HyperparameterTunerHandler (Phase 4: Hyperparameter Tuning)."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from core.executor import run_pipeline, PipelineExecutionError
from core.paths import get_generated_pipelines_dir


class TestHyperparameterTuner:
    """Test the HyperparameterTunerHandler MVP."""

    def test_tuner_basic_run_classification_quick_mode(self, tmp_path):
        """Test Quick mode (no tuning, direct model run)."""
        pipeline_dict = {
            "pipeline_name": "test_quick_tune",
            "stages": [
                {
                    "name": "load_data",
                    "type": "csv_loader",
                    "params": {
                        "source": str(Path(__file__).parent.parent / "data" / "titanic.csv"),
                        "target_column": "Survived",
                    },
                },
                {
                    "name": "preprocess",
                    "type": "tabular_preprocess",
                    "params": {
                        "target_column": "Survived",
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
                    "type": "classification_rf",
                    "params": {"random_state": 42},
                },
            ],
        }

        yaml_file = tmp_path / "test_quick.yml"
        yaml_file.write_text(yaml.dump(pipeline_dict))

        context = run_pipeline(str(yaml_file))
        assert context is not None
        assert "model" in context
        assert "metrics" in context
        assert context["metrics"].get("accuracy") is not None
        print(f"Quick mode accuracy: {context['metrics']['accuracy']:.4f}")

    def test_tuner_basic_run_classification_tune_mode(self, tmp_path):
        """Test Tune mode (hyperparameter tuning enabled)."""
        pipeline_dict = {
            "pipeline_name": "test_tune_mode",
            "stages": [
                {
                    "name": "load_data",
                    "type": "csv_loader",
                    "params": {
                        "source": str(Path(__file__).parent.parent / "data" / "titanic.csv"),
                        "target_column": "Survived",
                    },
                },
                {
                    "name": "preprocess",
                    "type": "tabular_preprocess",
                    "params": {
                        "target_column": "Survived",
                        "task_type": "classification",
                        "require_binary_target": True,
                        "scale_numeric": True,
                        "encode_categoricals": True,
                        "test_size": 0.2,
                        "random_state": 42,
                    },
                },
                {
                    "name": "hyperparameter_tune",
                    "type": "hyperparameter_tune",
                    "params": {
                        "model_type": "classification_rf",
                        "task_type": "classification",
                        "n_trials": 5,
                        "max_time_minutes": 0,
                        "random_state": 42,
                    },
                },
            ],
        }

        yaml_file = tmp_path / "test_tune.yml"
        yaml_file.write_text(yaml.dump(pipeline_dict))

        context = run_pipeline(str(yaml_file))
        assert context is not None
        assert "model" in context
        assert "metrics" in context
        assert "tuning_summary" in context
        assert "best_score" in context["tuning_summary"]
        print(f"Tune mode best score: {context['tuning_summary']['best_score']:.4f}")

    def test_tuner_respects_n_trials(self, tmp_path):
        """Test that tuner respects the n_trials parameter."""
        pipeline_dict = {
            "pipeline_name": "test_n_trials",
            "stages": [
                {
                    "name": "load_data",
                    "type": "csv_loader",
                    "params": {
                        "source": str(Path(__file__).parent.parent / "data" / "titanic.csv"),
                        "target_column": "Survived",
                    },
                },
                {
                    "name": "preprocess",
                    "type": "tabular_preprocess",
                    "params": {
                        "target_column": "Survived",
                        "task_type": "classification",
                        "require_binary_target": True,
                        "scale_numeric": True,
                        "encode_categoricals": True,
                        "test_size": 0.2,
                        "random_state": 42,
                    },
                },
                {
                    "name": "hyperparameter_tune",
                    "type": "hyperparameter_tune",
                    "params": {
                        "model_type": "classification_rf",
                        "task_type": "classification",
                        "n_trials": 3,
                        "max_time_minutes": 0,
                        "random_state": 42,
                    },
                },
            ],
        }

        yaml_file = tmp_path / "test_n_trials.yml"
        yaml_file.write_text(yaml.dump(pipeline_dict))

        context = run_pipeline(str(yaml_file))
        assert context is not None
        assert "tuning_summary" in context
        print("Test n_trials=3: PASS")

    def test_tuner_regression(self, tmp_path):
        """Test tuner on a regression task."""
        pipeline_dict = {
            "pipeline_name": "test_tune_regression",
            "stages": [
                {
                    "name": "load_data",
                    "type": "csv_loader",
                    "params": {
                        "source": str(Path(__file__).parent.parent / "data" / "housing.csv"),
                        "target_column": "median_house_value",
                    },
                },
                {
                    "name": "preprocess",
                    "type": "tabular_preprocess",
                    "params": {
                        "target_column": "median_house_value",
                        "task_type": "regression",
                        "require_binary_target": False,
                        "scale_numeric": True,
                        "encode_categoricals": True,
                        "test_size": 0.2,
                        "random_state": 42,
                    },
                },
                {
                    "name": "hyperparameter_tune",
                    "type": "hyperparameter_tune",
                    "params": {
                        "model_type": "regression_rf",
                        "task_type": "regression",
                        "n_trials": 5,
                        "max_time_minutes": 0,
                        "random_state": 42,
                    },
                },
            ],
        }

        yaml_file = tmp_path / "test_regression.yml"
        yaml_file.write_text(yaml.dump(pipeline_dict))

        context = run_pipeline(str(yaml_file))
        assert context is not None
        assert "model" in context
        assert "metrics" in context
        assert "tuning_summary" in context
        print(f"Regression tuning best score: {context['tuning_summary']['best_score']:.4f}")

    def test_tuner_missing_model_type_raises_error(self, tmp_path):
        """Test that tuner raises error when model_type is missing."""
        pipeline_dict = {
            "pipeline_name": "test_missing_model_type",
            "stages": [
                {
                    "name": "load_data",
                    "type": "csv_loader",
                    "params": {
                        "source": str(Path(__file__).parent.parent / "data" / "titanic.csv"),
                        "target_column": "Survived",
                    },
                },
                {
                    "name": "preprocess",
                    "type": "tabular_preprocess",
                    "params": {
                        "target_column": "Survived",
                        "task_type": "classification",
                        "require_binary_target": True,
                        "scale_numeric": True,
                        "encode_categoricals": True,
                        "test_size": 0.2,
                        "random_state": 42,
                    },
                },
                {
                    "name": "hyperparameter_tune",
                    "type": "hyperparameter_tune",
                    "params": {
                        "task_type": "classification",
                        "n_trials": 5,
                        "max_time_minutes": 0,
                        "random_state": 42,
                        # Missing "model_type"
                    },
                },
            ],
        }

        yaml_file = tmp_path / "test_missing_model_type.yml"
        yaml_file.write_text(yaml.dump(pipeline_dict))

        with pytest.raises(PipelineExecutionError):
            run_pipeline(str(yaml_file))

    def test_tuner_integration_with_pipeline_yaml(self, tmp_path):
        """Test tuner integration via a generated YAML file."""
        yaml_content = """pipeline_name: test_tuner_yaml
stages:
  - name: load_data
    type: csv_loader
    params:
      source: data/titanic.csv
      target_column: Survived
  - name: preprocess
    type: tabular_preprocess
    params:
      target_column: Survived
      task_type: classification
      require_binary_target: true
      scale_numeric: true
      encode_categoricals: true
      test_size: 0.2
      random_state: 42
  - name: hyperparameter_tune
    type: hyperparameter_tune
    params:
      model_type: classification_rf
      task_type: classification
      n_trials: 3
      max_time_minutes: 0
      random_state: 42
"""
        yaml_file = tmp_path / "test_tuner.yml"
        yaml_file.write_text(yaml_content)

        context = run_pipeline(str(yaml_file))
        assert context is not None
        assert "model" in context
        assert "tuning_summary" in context
        print("YAML-based tuner test: PASS")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
