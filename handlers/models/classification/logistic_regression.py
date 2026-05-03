from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from handlers.base import BaseHandler

class LogisticRegressionHandler(BaseHandler):
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        X_train = context.get("X_train")
        y_train = context.get("y_train")
        X_test = context.get("X_test")
        y_test = context.get("y_test")

        if X_train is None or y_train is None:
            raise ValueError("LogisticRegressionHandler: missing train data")

        # --- Hyperparameters ---
        # Default to some standard params
        C: float = 1.0
        max_iter: int = 100
        random_state: int = int(context.get("_random_seed", 42))
        class_weight = None  # None = no weighting, 'balanced' = auto-weight by inverse class freq
        
        # Parse params from YAML
        params = self.stage.params
        if "C" in params:
            C = float(params["C"])
        if "max_iter" in params:
            max_iter = int(params["max_iter"])
        if "random_state" in params:
            random_state = int(params["random_state"])
        if "class_weight" in params and params["class_weight"]:
            class_weight = "balanced"

        # --- Training ---
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state, class_weight=class_weight)
        model.fit(X_train, y_train)

        # --- Predictions ---
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="binary"))

        metrics = {"accuracy": acc, "f1": f1}
        
        # --- Visualization Artifacts ---
        artifacts = {}
        
        # 1. Confusion Matrix Data
        artifacts["y_test"] = y_test
        artifacts["y_pred"] = y_pred

        # 2. Feature Importance (Coefficients)
        # Handle binary (1 row) vs multiclass (n_classes rows)
        if hasattr(model, "coef_"):
            # Take absolute average of coefficients to represent magnitude/importance
            if model.coef_.ndim == 1:
                 artifacts["feature_importance"] = np.abs(model.coef_)
            else:
                 artifacts["feature_importance"] = np.mean(np.abs(model.coef_), axis=0)
            
            # Helper to get names
            if "preprocessor" in context:
                try:
                    artifacts["feature_names"] = context["preprocessor"].get_feature_names_out()
                except:
                    pass

        # 3. ROC Curve Data (Probabilities)
        if hasattr(model, "predict_proba"):
            artifacts["y_proba"] = model.predict_proba(X_test)
            artifacts["classes"] = model.classes_

        context["model"] = model
        context["y_pred"] = y_pred
        context["metrics"] = metrics
        context["artifacts"] = artifacts

        weight_mode = "balanced" if class_weight else "none"
        print(f"[logistic_regression] Trained LogReg (C={C}, class_weight={weight_mode}) -> acc={acc:.4f}")
        return context
