from __future__ import annotations
from typing import Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from handlers.base import BaseHandler

class LinearRegressionHandler(BaseHandler):
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        X_train, X_test = context["X_train"], context["X_test"]
        y_train, y_test = context["y_train"], context["y_test"]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = float(mean_squared_error(y_test, y_pred, squared=False))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        context["model"] = model
        context["metrics"] = {"rmse": rmse, "mae": mae, "r2": r2}
        print(f"[linear_reg] Trained LinearRegression → rmse={rmse:.4f}, r2={r2:.4f}")
        return context
