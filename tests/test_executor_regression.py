from core.executor import run_pipeline


def test_run_house_regression_pipeline_has_model_and_metrics() -> None:
    context = run_pipeline("examples/house_regression.yml")

    assert "model" in context
    assert "metrics" in context
    assert "artifacts" in context

    metrics = context["metrics"]
    assert "rmse" in metrics
    assert "r2" in metrics

    artifacts = context["artifacts"]
    assert "y_test" in artifacts
    assert "y_pred" in artifacts
