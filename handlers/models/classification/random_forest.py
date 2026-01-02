# handlers/models/classification/random_forest.py

from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from handlers.base import BaseHandler

class RandomForestTrainHandler(BaseHandler):
    """
    Trains a RandomForestClassifier and generates visualization artifacts.
    """
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        X_train = context.get("X_train")
        y_train = context.get("y_train")
        X_test = context.get("X_test")
        y_test = context.get("y_test")

        if X_train is None or y_train is None or X_test is None or y_test is None:
            raise ValueError("RandomForestTrainHandler: missing train/test data in context")

        # --- Hyperparameters ---
        n_estimators: int = 100
        max_depth: Optional[int] = None
        random_state = 42

        if self.stage.models:
            cfg = self.stage.models[0]
            n_estimators = int(cfg.get("n_estimators", n_estimators))
            if "max_depth" in cfg:
                max_depth = int(cfg["max_depth"])
        else:
            params = self.stage.params
            if "n_estimators" in params:
                n_estimators = int(params["n_estimators"])
            if "max_depth" in params:
                max_depth = int(params["max_depth"])

        # --- Training ---
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # --- Predictions ---
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="binary"))

        metrics = {"accuracy": acc, "f1": f1}
        
        # --- Visualization Artifacts (NEW) ---
        artifacts = {}
        
        # 1. For Confusion Matrix
        artifacts["y_test"] = y_test
        artifacts["y_pred"] = y_pred
        
        # 2. For Feature Importance
        if hasattr(model, "feature_importances_"):
            artifacts["feature_importance"] = model.feature_importances_
            # Try to get feature names from preprocessor
            if "preprocessor" in context:
                try:
                    # Attempt to get names from sklearn ColumnTransformer
                    artifacts["feature_names"] = context["preprocessor"].get_feature_names_out()
                except:
                    # Fallback
                    artifacts["feature_names"] = [f"Feat {i}" for i in range(len(model.feature_importances_))]

        # 3. For ROC Curve
        if hasattr(model, "predict_proba"):
            artifacts["y_proba"] = model.predict_proba(X_test)
            artifacts["classes"] = model.classes_

        # Update Context
        context["model"] = model
        context["y_pred"] = y_pred
        context["metrics"] = metrics
        context["artifacts"] = artifacts  # Store for GUI

        print(
            f"[random_forest] Trained RF (n={n_estimators}, depth={max_depth}) → "
            f"accuracy={acc:.4f}, f1={f1:.4f}"
        )

        return context
