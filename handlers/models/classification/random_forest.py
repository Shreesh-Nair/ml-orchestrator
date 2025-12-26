# handlers/models/classification/random_forest.py
from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from handlers.base import BaseHandler


class RandomForestTrainHandler(BaseHandler):
    """
    Trains a RandomForestClassifier on X_train/y_train.
    Reads hyperparameters from:
      - stage.models[0] (if present), or
      - stage.params
    Stores:
      - context["model"]
      - context["y_pred"]
      - context["metrics"]
    """

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        X_train = context.get("X_train")
        y_train = context.get("y_train")
        X_test = context.get("X_test")
        y_test = context.get("y_test")

        if X_train is None or y_train is None or X_test is None or y_test is None:
            raise ValueError("RandomForestTrainHandler: missing train/test data in context")

        # Default hyperparameters
        n_estimators: int = 100
        max_depth: Optional[int] = None
        random_state = 42

        # Prefer models block if present
        if self.stage.models:
            first_model_cfg = self.stage.models[0]
            n_estimators = int(first_model_cfg.get("n_estimators", n_estimators))
            if "max_depth" in first_model_cfg:
                max_depth = int(first_model_cfg["max_depth"])
        else:
            # Fallback to params
            params = self.stage.params
            if "n_estimators" in params:
                n_estimators = int(params["n_estimators"])
            if "max_depth" in params:
                max_depth = int(params["max_depth"])

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="binary"))

        metrics = {
            "accuracy": acc,
            "f1": f1,
        }

        context["model"] = model
        context["y_pred"] = y_pred
        context["metrics"] = metrics

        print(
            f"[random_forest] Trained RandomForest (n_estimators={n_estimators}, max_depth={max_depth}) → "
            f"accuracy={acc:.4f}, f1={f1:.4f}"
        )

        return context
