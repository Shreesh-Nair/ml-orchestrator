# tests/test_executor_fraud.py
from core.executor import run_pipeline


def test_run_fraud_anomaly_pipeline_has_anomaly_metrics():
    context = run_pipeline("examples/fraud_anomaly.yml")

    assert "anomaly_model" in context
    assert "anomaly_metrics" in context
    m = context["anomaly_metrics"]
    assert "auc" in m
    assert "precision" in m
    assert "recall" in m
    assert "f1" in m

    assert "artifacts" in context
    artifacts = context["artifacts"]
    assert "y_test" in artifacts
    assert "anomaly_scores" in artifacts
    assert "anomaly_preds" in artifacts
