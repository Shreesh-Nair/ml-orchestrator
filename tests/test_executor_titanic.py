# tests/test_executor_titanic.py
from core.executor import run_pipeline


def test_run_titanic_pipeline_has_model_and_metrics():
    context = run_pipeline("examples/titanic.yml")

    assert "model" in context
    assert "metrics" in context
    m = context["metrics"]
    assert "accuracy" in m
    assert "f1" in m
