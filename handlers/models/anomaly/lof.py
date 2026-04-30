from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from handlers.base import BaseHandler


class LOFHandler(BaseHandler):
    """
    Trains a Local Outlier Factor anomaly detector and generates evaluation artifacts.
    """

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        X_train = context.get("X_train")
        y_train = context.get("y_train")
        X_test = context.get("X_test")
        y_test = context.get("y_test")

        if X_train is None or y_train is None or X_test is None or y_test is None:
            raise ValueError("LOFHandler: missing train/test data in context")

        # Hyperparameters
        n_neighbors: int = 20
        contamination: float = "auto"

        if self.stage.models:
            cfg = self.stage.models[0]
            n_neighbors = int(cfg.get("n_neighbors", n_neighbors))
            contamination = cfg.get("contamination", contamination)
        else:
            params = self.stage.params
            if "n_neighbors" in params:
                n_neighbors = int(params["n_neighbors"])
            if "contamination" in params:
                contamination = params["contamination"]

        # Train on full training set (LOF can use semi-supervised approach)
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
        )
        lof.fit(X_train)

        # Predictions on test set
        y_pred = lof.predict(X_test)  # -1 for anomalies, 1 for normal
        anomaly_scores = -lof.score_samples(X_test)  # Negative LOF scores for interpretability

        # Convert to binary classification: -1 -> 1 (anomaly), 1 -> 0 (normal)
        y_pred_binary = (y_pred == -1).astype(int)

        # Metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
        precision = float(precision_score(y_test, y_pred_binary, zero_division=0))
        recall = float(recall_score(y_test, y_pred_binary, zero_division=0))
        f1 = float(f1_score(y_test, y_pred_binary, zero_division=0))

        metrics = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        # Artifacts
        artifacts = {}
        artifacts["y_test"] = np.asarray(y_test)
        artifacts["y_pred"] = np.asarray(y_pred_binary)
        artifacts["anomaly_scores"] = np.asarray(anomaly_scores)

        context["model"] = lof
        context["metrics"] = metrics
        context["artifacts"] = artifacts

        print(f"[anomaly_lof] Trained LOF -> precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
        return context
