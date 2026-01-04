from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from handlers.base import BaseHandler

class RandomForestRegressionHandler(BaseHandler):
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        X_train = context.get("X_train")
        y_train = context.get("y_train")
        X_test = context.get("X_test")
        y_test = context.get("y_test")

        if X_train is None or y_train is None:
            raise ValueError("RandomForestRegressorHandler: missing data")

        # --- Hyperparameters ---
        n_estimators: int = 100
        max_depth: Optional[int] = None
        
        params = self.stage.params
        if "n_estimators" in params:
            n_estimators = int(params["n_estimators"])
        if "max_depth" in params:
            max_depth = int(params["max_depth"])

        # --- Training ---
        model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # --- Metrics ---
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))

        metrics = {"rmse": rmse, "r2": r2}
        
        # --- Visualization Artifacts ---
        artifacts = {}
        
        # 1. Feature Importance (Supported by RF Regressor)
        if hasattr(model, "feature_importances_"):
            artifacts["feature_importance"] = model.feature_importances_
            
            if "preprocessor" in context:
                try:
                    artifacts["feature_names"] = context["preprocessor"].get_feature_names_out()
                except:
                    pass

        # Note: No CM or ROC for regression.
        
        context["model"] = model
        context["metrics"] = metrics
        context["artifacts"] = artifacts

        print(f"[regression_rf] Trained RF Regressor -> rmse={rmse:.4f}, r2={r2:.4f}")
        return context
