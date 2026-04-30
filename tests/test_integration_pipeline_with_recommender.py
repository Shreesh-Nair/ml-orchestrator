# tests/test_integration_pipeline_with_recommender.py
"""
Integration tests for complete pipeline flow:
  1. Load data
  2. Analyze data quality & generate recommendations
  3. Generate YAML with recommended preprocess params
  4. Train model via executor
  5. Make predictions
  6. Validate results
"""

import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from core.data_quality import analyze_data_quality, recommend_quick_fixes
from core.executor import run_pipeline


def test_housing_regression_with_recommender():
    """
    Test housing regression pipeline with recommender-driven params:
    - Load housing data
    - Get recommendations
    - Generate YAML with preprocess params
    - Train + validate
    """
    csv_path = "data/housing.csv"
    df = pd.read_csv(csv_path)
    target_col = "median_house_value"
    
    # Analyze data quality and get recommendations
    report = analyze_data_quality(df, target_column=target_col)
    recommendations = recommend_quick_fixes(report)
    preprocess_params = recommendations.get("preprocess_params", {})
    
    # Build preprocess stage params with recommendations
    preprocess_stage_params = {
        "impute_missing": True,
        "scale_numeric": True,
        "task_type": "regression",  # Required for regression tasks
    }
    preprocess_stage_params.update(preprocess_params)
    
    # Build pipeline YAML matching handler registry structure
    pipeline_yaml = {
        "pipeline_name": "housing_regression_recommended",
        "stages": [
            {
                "name": "load_data",
                "type": "csv_loader",
                "params": {
                    "source": csv_path,
                    "target_column": target_col,
                },
            },
            {
                "name": "preprocess",
                "type": "tabular_preprocess",
                "params": preprocess_stage_params,
            },
            {
                "name": "train",
                "type": "regression",
                "params": {
                    "n_estimators": 10,
                },
            },
        ],
    }
    
    yaml_content = yaml.dump(pipeline_yaml, default_flow_style=False)
    
    # Write YAML to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name
    
    try:
        # Run pipeline
        context = run_pipeline(yaml_path)
        
        # Validate outputs
        assert "model" in context, "Expected 'model' in context"
        assert "metrics" in context, "Expected 'metrics' in context"
        
        metrics = context["metrics"]
        assert "rmse" in metrics, "Expected RMSE in metrics"
        assert "r2" in metrics, "Expected R² in metrics"
        
        # Validate training data was loaded
        assert "artifacts" in context, "Expected artifacts in context"
        
    finally:
        Path(yaml_path).unlink()


def test_titanic_classification_with_recommender():
    """
    Test titanic classification pipeline with recommender-driven params:
    - Load titanic data
    - Get recommendations (date extraction, rare category grouping)
    - Generate YAML with preprocess params
    - Train + validate
    """
    csv_path = "data/titanic.csv"
    df = pd.read_csv(csv_path)
    target_col = "Survived"
    
    # Analyze data quality and get recommendations
    report = analyze_data_quality(df, target_column=target_col)
    recommendations = recommend_quick_fixes(report)
    preprocess_params = recommendations.get("preprocess_params", {})
    
    # Build preprocess stage params with recommendations
    preprocess_stage_params = {
        "impute_missing": True,
        "scale_numeric": True,
    }
    preprocess_stage_params.update(preprocess_params)
    
    # Build pipeline YAML
    pipeline_yaml = {
        "pipeline_name": "titanic_classification_recommended",
        "stages": [
            {
                "name": "load_data",
                "type": "csv_loader",
                "params": {
                    "source": csv_path,
                    "target_column": target_col,
                },
            },
            {
                "name": "preprocess",
                "type": "tabular_preprocess",
                "params": preprocess_stage_params,
            },
            {
                "name": "train",
                "type": "classification",
                "params": {
                    "n_estimators": 10,
                },
            },
        ],
    }
    
    yaml_content = yaml.dump(pipeline_yaml, default_flow_style=False)
    
    # Write YAML to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name
    
    try:
        # Run pipeline
        context = run_pipeline(yaml_path)
        
        # Validate outputs
        assert "model" in context, "Expected 'model' in context"
        assert "metrics" in context, "Expected 'metrics' in context"
        
        metrics = context["metrics"]
        assert "accuracy" in metrics, "Expected accuracy in metrics"
        assert "f1" in metrics, "Expected F1 in metrics"
        assert 0 <= metrics["accuracy"] <= 1, "Accuracy should be in [0, 1]"
        
        # Validate training data was loaded
        assert "artifacts" in context, "Expected artifacts in context"
        
    finally:
        Path(yaml_path).unlink()


def test_fraud_anomaly_with_recommender():
    """
    Test fraud anomaly detection with recommender-driven params:
    - Load fraud data
    - Get recommendations
    - Generate YAML with preprocess params
    - Train + validate
    """
    csv_path = "data/fraud.csv"
    df = pd.read_csv(csv_path)
    target_col = "Class"
    
    # Analyze data quality and get recommendations
    report = analyze_data_quality(df, target_column=target_col)
    recommendations = recommend_quick_fixes(report)
    preprocess_params = recommendations.get("preprocess_params", {})
    
    preprocess_stage_params = {
        "impute_missing": True,
        "scale_numeric": True,
    }
    preprocess_stage_params.update(preprocess_params)
    
    # Build pipeline YAML for anomaly detection
    pipeline_yaml = {
        "pipeline_name": "fraud_anomaly_recommended",
        "stages": [
            {
                "name": "load_data",
                "type": "csv_loader",
                "params": {
                    "source": csv_path,
                    "target_column": target_col,
                },
            },
            {
                "name": "preprocess",
                "type": "tabular_preprocess",
                "params": preprocess_stage_params,
            },
            {
                "name": "train",
                "type": "anomaly_isolation_forest",
                "params": {
                    "contamination": 0.01,
                    "n_estimators": 10,
                },
            },
        ],
    }
    
    yaml_content = yaml.dump(pipeline_yaml, default_flow_style=False)
    
    # Write YAML to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name
    
    try:
        # Run pipeline
        context = run_pipeline(yaml_path)
        
        # Validate outputs
        assert "anomaly_model" in context, "Expected 'anomaly_model' in context"
        assert "anomaly_metrics" in context, "Expected 'anomaly_metrics' in context"
        
        metrics = context["anomaly_metrics"]
        assert "auc" in metrics, "Expected AUC in metrics"
        assert "precision" in metrics, "Expected precision in metrics"
        
        # Validate artifacts
        assert "artifacts" in context, "Expected artifacts in context"
        
    finally:
        Path(yaml_path).unlink()


def test_recommender_respects_toggles():
    """
    Test that recommender params can be selectively enabled/disabled:
    - Generate recommendations for titanic
    - Create two pipelines: one with all params, one with subset
    - Verify both train successfully with different preprocess configs
    """
    csv_path = "data/titanic.csv"
    df = pd.read_csv(csv_path)
    target_col = "Survived"
    
    # Get full recommendations
    report = analyze_data_quality(df, target_column=target_col)
    recommendations = recommend_quick_fixes(report)
    full_preprocess_params = recommendations.get("preprocess_params", {})
    
    # Create YAML with all recommendations
    full_yaml = {
        "pipeline_name": "titanic_full_params",
        "stages": [
            {
                "name": "load_data",
                "type": "csv_loader",
                "params": {
                    "source": csv_path,
                    "target_column": target_col,
                },
            },
            {
                "name": "preprocess",
                "type": "tabular_preprocess",
                "params": {
                    "impute_missing": True,
                    "scale_numeric": True,
                    **full_preprocess_params,
                },
            },
            {
                "name": "train",
                "type": "classification",
                "params": {
                    "n_estimators": 10,
                },
            },
        ],
    }
    
    # Create YAML with subset of recommendations (only date extraction if available)
    subset_preprocess_params = {}
    if "date_extract" in full_preprocess_params:
        subset_preprocess_params["date_extract"] = True
    
    subset_yaml = {
        "pipeline_name": "titanic_subset_params",
        "stages": [
            {
                "name": "load_data",
                "type": "csv_loader",
                "params": {
                    "source": csv_path,
                    "target_column": target_col,
                },
            },
            {
                "name": "preprocess",
                "type": "tabular_preprocess",
                "params": {
                    "impute_missing": True,
                    "scale_numeric": True,
                    **subset_preprocess_params,
                },
            },
            {
                "name": "train",
                "type": "classification",
                "params": {
                    "n_estimators": 10,
                },
            },
        ],
    }
    
    full_yaml_str = yaml.dump(full_yaml, default_flow_style=False)
    subset_yaml_str = yaml.dump(subset_yaml, default_flow_style=False)
    
    # Test full params pipeline
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(full_yaml_str)
        full_yaml_path = f.name
    
    try:
        context_full = run_pipeline(full_yaml_path)
        assert "model" in context_full
        assert "metrics" in context_full
    finally:
        Path(full_yaml_path).unlink()
    
    # Test subset params pipeline
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(subset_yaml_str)
        subset_yaml_path = f.name
    
    try:
        context_subset = run_pipeline(subset_yaml_path)
        assert "model" in context_subset
        assert "metrics" in context_subset
    finally:
        Path(subset_yaml_path).unlink()
    
    # Both should train successfully (may have different metrics)
    assert context_full["metrics"]["accuracy"] >= 0, "Full pipeline should produce valid metrics"
    assert context_subset["metrics"]["accuracy"] >= 0, "Subset pipeline should produce valid metrics"
