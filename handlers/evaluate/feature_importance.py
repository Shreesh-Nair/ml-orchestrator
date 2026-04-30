from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.inspection import permutation_importance

from handlers.base import BaseHandler


class FeatureImportanceHandler(BaseHandler):
    """
    Computes feature importance and explainability metrics for trained models.
    Supports both built-in feature_importances_ and permutation importance.
    """

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        model = context.get("model")
        if model is None:
            raise ValueError("FeatureImportanceHandler: no trained model in context")

        X_test = context.get("X_test")
        y_test = context.get("y_test")
        if X_test is None or y_test is None:
            raise ValueError("FeatureImportanceHandler: missing X_test or y_test")

        # Get artifacts dict (create if not exists)
        artifacts = context.get("artifacts", {})

        # Method 1: Built-in feature importances (for tree-based models)
        if hasattr(model, "feature_importances_"):
            built_in_importance = model.feature_importances_
            artifacts["feature_importance"] = built_in_importance
            artifacts["importance_type"] = "built_in"

        # Method 2: Coefficient-based importance (for linear models)
        elif hasattr(model, "coef_"):
            coef_importance = np.abs(model.coef_)
            if coef_importance.ndim > 1:
                coef_importance = coef_importance.ravel()
            artifacts["feature_importance"] = coef_importance
            artifacts["importance_type"] = "coefficient"

        # Method 3: Permutation importance (model-agnostic)
        try:
            task = context.get("_task_type", "classification")
            scoring = "accuracy" if task == "classification" else "neg_mean_squared_error"
            
            perm_importance = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=10,
                random_state=42,
                n_jobs=-1,
                scoring=scoring,
            )
            artifacts["permutation_importance"] = perm_importance.importances_mean
            artifacts["permutation_importance_std"] = perm_importance.importances_std
            artifacts["importance_type"] = "permutation"
        except Exception as e:
            # Permutation importance can fail for some models; log but don't fail
            print(f"[feature_importance] Could not compute permutation importance: {e}")

        context["artifacts"] = artifacts
        print(f"[feature_importance] Computed feature importance with {artifacts.get('importance_type', 'unknown')} method")
        return context
