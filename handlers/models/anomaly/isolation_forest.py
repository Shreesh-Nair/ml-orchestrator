# handlers/models/anomaly/isolation_forest.py
from __future__ import annotations

from typing import Dict, Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from handlers.base import BaseHandler


class IsolationForestHandler(BaseHandler):
    """
    Runs IsolationForest anomaly detection on tabular data.

    Expects in context (from tabular_preprocess):
      - X_train, X_test: array-like
      - y_test: binary labels (0 = normal, 1 = anomaly/fraud)

    Parameters (stage.params):
      - contamination: float, expected fraction of anomalies in the data
      - random_state: int, random seed

    Adds to context:
      - context["anomaly_model"]
      - context["anomaly_scores"]
      - context["anomaly_preds"]
      - context["anomaly_metrics"]
    """

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        X_train = context.get("X_train")
        X_test = context.get("X_test")
        y_test = context.get("y_test")

        if X_train is None or X_test is None or y_test is None:
            raise ValueError("IsolationForestHandler: missing X_train/X_test/y_test in context")

        contamination = float(self.stage.params.get("contamination", 0.01))
        random_state = int(self.stage.params.get("random_state", context.get("_random_seed", 42)))

        model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            n_jobs=-1,
        )

        # Fit on train data only (unsupervised)
        model.fit(X_train)

        # Decision_function gives anomaly scores (higher = more normal), so invert
        scores = -model.decision_function(X_test)  # higher score = more anomalous
        # Predict: -1 = anomaly, 1 = normal
        preds_raw = model.predict(X_test)
        # Convert to 1 = anomaly, 0 = normal
        preds = np.where(preds_raw == -1, 1, 0)

        # Ensure y_test is numpy array
        y_true = np.asarray(y_test)

        # Basic metrics
        try:
            auc = float(roc_auc_score(y_true, scores))
        except ValueError:
            # AUC undefined if only one class present
            auc = float("nan")

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0
        )

        metrics = {
            "auc": auc,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        context["anomaly_model"] = model
        context["anomaly_scores"] = scores
        context["anomaly_preds"] = preds
        context["anomaly_metrics"] = metrics

        print(
            "[isolation_forest] contamination="
            f"{contamination}, AUC={auc:.4f}, precision={precision:.4f}, "
            f"recall={recall:.4f}, f1={f1:.4f}"
        )

        return context
