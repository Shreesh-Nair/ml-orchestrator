from __future__ import annotations
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from handlers.base import BaseHandler

class LogisticRegressionHandler(BaseHandler):
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        X_train, X_test = context["X_train"], context["X_test"]
        y_train, y_test = context["y_train"], context["y_test"]

        # Increase max_iter for convergence
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        context["model"] = model
        context["metrics"] = {"accuracy": acc, "f1": f1}
        print(f"[logistic_reg] Trained LogisticRegression → accuracy={acc:.4f}, f1={f1:.4f}")
        return context
