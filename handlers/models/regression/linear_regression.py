from __future__ import annotations
from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from handlers.base import BaseHandler

class LinearRegressionHandler(BaseHandler):
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        X_train = context.get("X_train")
        y_train = context.get("y_train")
        X_test = context.get("X_test")
        y_test = context.get("y_test")

        if X_train is None or y_train is None:
            raise ValueError("LinearRegressionHandler: missing data")

        # --- Training ---
        model = LinearRegression()
        model.fit(X_train, y_train)

        # --- Metrics ---
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))

        metrics = {"rmse": rmse, "r2": r2}

        # --- Visualization Artifacts ---
        artifacts = {}

        # 1. Feature Importance (Using absolute Coefficients)
        if hasattr(model, "coef_"):
            artifacts["feature_importance"] = np.abs(model.coef_)
            
            if "preprocessor" in context:
                try:
                    artifacts["feature_names"] = context["preprocessor"].get_feature_names_out()
                except:
                    pass

        context["model"] = model
        context["metrics"] = metrics
        context["artifacts"] = artifacts

        print(f"[regression_linear] Trained LinearRegression -> rmse={rmse:.4f}, r2={r2:.4f}")
        return context
