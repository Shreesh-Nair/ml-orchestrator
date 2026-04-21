from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from handlers.base import BaseHandler


class RandomForestTrainHandler(BaseHandler):
    """
    Trains a RandomForestClassifier and generates evaluation artifacts.
    """

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        X_train = context.get("X_train")
        y_train = context.get("y_train")
        X_test = context.get("X_test")
        y_test = context.get("y_test")

        if X_train is None or y_train is None or X_test is None or y_test is None:
            raise ValueError("RandomForestTrainHandler: missing train/test data in context")

        # Hyperparameters
        n_estimators: int = 100
        max_depth: Optional[int] = None
        random_state = 42

        if self.stage.models:
            cfg = self.stage.models[0]
            n_estimators = int(cfg.get("n_estimators", n_estimators))
            if "max_depth" in cfg and cfg["max_depth"] is not None:
                max_depth = int(cfg["max_depth"])
        else:
            params = self.stage.params
            if "n_estimators" in params:
                n_estimators = int(params["n_estimators"])
            if "max_depth" in params and params["max_depth"] is not None:
                max_depth = int(params["max_depth"])

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        classes = list(model.classes_)
        if len(classes) != 2:
            raise ValueError(
                "RandomForestTrainHandler: this MVP supports binary classification only. "
                f"Found classes: {classes}"
            )

        positive_label = classes[1]
        positive_index = classes.index(positive_label)

        y_true_arr = np.asarray(y_test)
        y_pred_arr = np.asarray(y_pred)
        y_true_bin = (y_true_arr == positive_label).astype(int)
        y_pred_bin = (y_pred_arr == positive_label).astype(int)
        y_score = y_proba[:, positive_index]

        metrics = {
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
            "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
            "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true_bin, y_score)),
        }

        artifacts: Dict[str, Any] = {
            "y_test": y_true_arr,
            "y_pred": y_pred_arr,
            "y_proba": y_proba,
            "classes": np.asarray(classes),
            "positive_label": positive_label,
        }

        if hasattr(model, "feature_importances_"):
            artifacts["feature_importance"] = model.feature_importances_
            feature_count = len(model.feature_importances_)
            feature_names = [f"feature_{idx}" for idx in range(feature_count)]
            preprocessor = context.get("preprocessor")
            if preprocessor is not None:
                try:
                    feature_names = preprocessor.get_feature_names_out().tolist()
                except Exception:
                    pass
            artifacts["feature_names"] = feature_names

        context["model"] = model
        context["y_pred"] = y_pred_arr
        context["metrics"] = metrics
        context["artifacts"] = artifacts

        print(
            f"[random_forest] Trained RF (n={n_estimators}, depth={max_depth}) -> "
            f"accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, auc={metrics['roc_auc']:.4f}"
        )

        return context
