from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - fallback keeps the app usable without xgboost installed
    from sklearn.ensemble import HistGradientBoostingClassifier as XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from handlers.base import BaseHandler


class GradientBoostingHandler(BaseHandler):
    """
    Trains an XGBoost classifier and generates evaluation artifacts.
    """

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        X_train = context.get("X_train")
        y_train = context.get("y_train")
        X_test = context.get("X_test")
        y_test = context.get("y_test")

        if X_train is None or y_train is None or X_test is None or y_test is None:
            raise ValueError("GradientBoostingHandler: missing train/test data in context")

        # Hyperparameters with sensible defaults
        n_estimators: int = 100
        max_depth: int = 6
        learning_rate: float = 0.1
        random_state = int(context.get("_random_seed", 42))

        if self.stage.models:
            cfg = self.stage.models[0]
            n_estimators = int(cfg.get("n_estimators", n_estimators))
            max_depth = int(cfg.get("max_depth", max_depth))
            learning_rate = float(cfg.get("learning_rate", learning_rate))
        else:
            params = self.stage.params
            if "random_state" in params:
                random_state = int(params["random_state"])
            if "n_estimators" in params:
                n_estimators = int(params["n_estimators"])
            if "max_depth" in params:
                max_depth = int(params["max_depth"])
            if "learning_rate" in params:
                learning_rate = float(params["learning_rate"])

        if XGBClassifier.__module__.startswith("xgboost"):
            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                eval_metric="logloss",
                verbosity=0,
            )
        else:
            model = XGBClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                max_iter=n_estimators,
                random_state=random_state,
            )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        classes = list(model.classes_)
        if len(classes) != 2:
            raise ValueError(
                "GradientBoostingHandler: this MVP supports binary classification only. "
                f"Found classes: {classes}"
            )

        # Metrics
        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_proba[:, 1]))

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        }

        # Artifacts
        artifacts = {}
        artifacts["y_test"] = np.asarray(y_test)
        artifacts["y_pred"] = np.asarray(y_pred)
        artifacts["y_proba"] = np.asarray(y_proba)
        artifacts["classes"] = classes

        # Feature importance
        if hasattr(model, "feature_importances_"):
            artifacts["feature_importance"] = model.feature_importances_

        context["model"] = model
        context["metrics"] = metrics
        context["artifacts"] = artifacts

        print(f"[classification_xgboost] Trained XGBoost -> accuracy={accuracy:.4f}, f1={f1:.4f}, roc_auc={roc_auc:.4f}")
        return context
