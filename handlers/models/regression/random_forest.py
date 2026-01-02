# handlers/models/regression/random_forest.py
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from handlers.base import BaseHandler


class RandomForestRegressionHandler(BaseHandler):
    """Train a RandomForest regressor on tabular data."""

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        X_train = context["X_train"]
        X_test = context["X_test"]
        y_train = context["y_train"]
        y_test = context["y_test"]

        params = self.stage.params or {}
        n_estimators = int(params.get("n_estimators", 100))
        max_depth = params.get("max_depth", None)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        rmse = float(mean_squared_error(y_test, y_pred, squared=False))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        context["model"] = model
        context["metrics"] = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

        print(
            f"[random_forest_reg] Trained RandomForestRegressor "
            f"(n_estimators={n_estimators}, max_depth={max_depth}) "
            f"→ rmse={rmse:.4f}, mae={mae:.4f}, r2={r2:.4f}"
        )

        return context
