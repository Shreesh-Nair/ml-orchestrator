# tests/test_performance_benchmark.py
"""
Performance benchmarks for preprocessing and training.

These tests measure execution time for common operations and help identify
performance regressions. Thresholds are conservative (generous) to allow
for variance in test machine specs.
"""

import time
from typing import Tuple

import pandas as pd
import pytest

from core.executor import run_pipeline
from handlers.preprocess.tabular_preprocess import TabularPreprocessHandler
from handlers.models.regression.random_forest import RandomForestRegressionHandler
from handlers.base import BaseHandler
from core.yaml_parser import Stage


def _create_large_dataframe(rows: int = 5000, cols: int = 20) -> Tuple[pd.DataFrame, str]:
    """Create a large synthetic dataset for benchmarking."""
    import numpy as np
    
    data = {
        f"feature_{i}": np.random.randn(rows) if i % 3 == 0 
                        else np.random.choice(['A', 'B', 'C', 'D'], rows)
        for i in range(cols - 1)
    }
    data["target"] = np.random.randn(rows)
    
    df = pd.DataFrame(data)
    return df, "target"


@pytest.mark.benchmark
def test_preprocess_performance_large_dataset():
    """
    Benchmark: Preprocessing a large dataset (5000 rows, 20 cols).
    
    Expected time: < 2 seconds
    Threshold: 5 seconds (2.5x safety margin)
    """
    df, target_col = _create_large_dataframe(rows=5000, cols=20)
    
    # Create a preprocess stage config
    stage = Stage(
        name="preprocess",
        type="tabular_preprocess",
        params={
            "impute_missing": True,
            "scale_numeric": True,
            "date_extract": False,
            "text_extract": False,
        }
    )
    
    handler = TabularPreprocessHandler(stage)
    context = {
        "df": df,
        "target_column": target_col,
    }
    
    # Measure preprocessing time
    start = time.time()
    result = handler.run(context)
    elapsed = time.time() - start
    
    assert "df_preprocessed" in result
    assert result["df_preprocessed"].shape[0] > 0
    assert elapsed < 5.0, f"Preprocessing took {elapsed:.2f}s, expected < 5s"
    
    print(f"\n✓ Preprocessing benchmark: {elapsed:.3f}s for {df.shape} dataset")


@pytest.mark.benchmark
def test_training_performance_medium_dataset():
    """
    Benchmark: Training a RandomForest on medium dataset (1000 rows, 15 cols).
    
    Expected time: < 3 seconds
    Threshold: 10 seconds (3.3x safety margin)
    """
    df, target_col = _create_large_dataframe(rows=1000, cols=15)
    
    # Simulate preprocessing output (already transformed)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Create a training stage config
    stage = Stage(
        name="train",
        type="regression",
        params={
            "n_estimators": 50,
            "max_depth": 10,
        }
    )
    
    handler = RandomForestRegressionHandler(stage)
    context = {
        "X_train": X,
        "y_train": y,
    }
    
    # Measure training time
    start = time.time()
    result = handler.run(context)
    elapsed = time.time() - start
    
    assert "model" in result
    assert result["model"] is not None
    assert elapsed < 10.0, f"Training took {elapsed:.2f}s, expected < 10s"
    
    print(f"\n✓ Training benchmark: {elapsed:.3f}s for {X.shape} dataset")


@pytest.mark.benchmark
def test_full_pipeline_performance():
    """
    Benchmark: Full end-to-end pipeline on housing dataset.
    
    Expected time: < 5 seconds
    Threshold: 15 seconds (3x safety margin)
    """
    csv_path = "data/housing.csv"
    df = pd.read_csv(csv_path)
    target_col = "median_house_value"
    
    # Create a minimal YAML pipeline
    import yaml
    import tempfile
    from pathlib import Path
    
    pipeline_yaml = {
        "pipeline_name": "housing_perf_test",
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
                },
            },
            {
                "name": "train",
                "type": "regression",
                "params": {
                    "n_estimators": 10,
                    "max_depth": 5,
                },
            },
        ],
    }
    
    yaml_content = yaml.dump(pipeline_yaml, default_flow_style=False)
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name
    
    try:
        # Measure full pipeline time
        start = time.time()
        context = run_pipeline(yaml_path)
        elapsed = time.time() - start
        
        assert "model" in context
        assert "metrics" in context
        assert elapsed < 15.0, f"Full pipeline took {elapsed:.2f}s, expected < 15s"
        
        print(f"\n✓ Full pipeline benchmark: {elapsed:.3f}s")
        
    finally:
        Path(yaml_path).unlink()


@pytest.mark.benchmark
def test_recommender_analysis_performance():
    """
    Benchmark: Data quality analysis + recommender on large dataset.
    
    Expected time: < 1 second
    Threshold: 3 seconds (3x safety margin)
    """
    from core.data_quality import analyze_data_quality, recommend_quick_fixes
    
    df, _ = _create_large_dataframe(rows=5000, cols=20)
    target_col = df.columns[-1]
    
    # Measure analysis + recommendation time
    start = time.time()
    report = analyze_data_quality(df, target_column=target_col)
    recommendations = recommend_quick_fixes(report)
    elapsed = time.time() - start
    
    assert "summary" in report
    assert "preprocess_params" in recommendations
    assert elapsed < 3.0, f"Analysis took {elapsed:.2f}s, expected < 3s"
    
    print(f"\n✓ Data quality analysis benchmark: {elapsed:.3f}s for {df.shape} dataset")


def test_benchmark_summary(benchmark=None):
    """
    Summary: Print benchmark thresholds for reference.
    """
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARKS - Thresholds")
    print("=" * 70)
    print("Preprocessing (5K rows, 20 cols):     < 5.0 seconds")
    print("Training (1K rows, 15 cols):          < 10.0 seconds")
    print("Full pipeline (housing dataset):      < 15.0 seconds")
    print("Data quality analysis (5K rows):      < 3.0 seconds")
    print("=" * 70)
    print("\nThresholds are conservative (3x typical execution time)")
    print("to allow for variance across different test environments.\n")
