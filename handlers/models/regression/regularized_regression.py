from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from handlers.base import BaseHandler


class RegularizedRegressionHandler(BaseHandler):
    """
    Trains regularized regression models (Ridge, Lasso, ElasticNet) and generates evaluation artifacts.
    """

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        X_train = context.get("X_train")
        y_train = context.get("y_train")
        X_test = context.get("X_test")
        y_test = context.get("y_test")

        if X_train is None or y_train is None:
            raise ValueError("RegularizedRegressionHandler: missing train/test data in context")

        # Determine which regularization type to use
        model_type = self.stage.type if hasattr(self.stage, "type") else "regression_ridge"
        
        # Default hyperparameters
        alpha: float = 1.0
        l1_ratio: float = 0.5  # Only used for ElasticNet
        random_state = int(context.get("_random_seed", 42))

        if self.stage.models:
            cfg = self.stage.models[0]
            alpha = float(cfg.get("alpha", alpha))
            if "l1_ratio" in cfg:
                l1_ratio = float(cfg["l1_ratio"])
        else:
            params = self.stage.params
            if "alpha" in params:
                alpha = float(params["alpha"])
            if "l1_ratio" in params:
                l1_ratio = float(params["l1_ratio"])

        # Instantiate appropriate model
        if "lasso" in model_type.lower():
            model = Lasso(alpha=alpha, random_state=random_state, max_iter=10000)
        elif "elasticnet" in model_type.lower():
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=10000)
        else:  # Default to Ridge
            model = Ridge(alpha=alpha, random_state=random_state)

        # Training
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        metrics = {"rmse": rmse, "mae": mae, "r2": r2}

        # Artifacts
        artifacts = {}
        artifacts["y_test"] = np.asarray(y_test)
        artifacts["y_pred"] = np.asarray(y_pred)

        # Feature importance (absolute coefficients)
        if hasattr(model, "coef_"):
            artifacts["feature_importance"] = np.abs(model.coef_)

        context["model"] = model
        context["metrics"] = metrics
        context["artifacts"] = artifacts

        model_name = model_type.replace("regression_", "").upper()
        print(f"[regression_regularized] Trained {model_name} (alpha={alpha}) -> rmse={rmse:.4f}, r2={r2:.4f}")
        return context
