from __future__ import annotations

from typing import Any, Dict
import copy
import random
import time

import numpy as np

from handlers.base import BaseHandler
from core.handler_registry import get_handler_for_stage


class HyperparameterTunerHandler(BaseHandler):
    """Simple hyperparameter tuner that delegates training to existing model handlers.

    It performs randomized sampling over small predefined search spaces for known model types.
    """

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        params = self.stage.params or {}
        model_type = params.get("model_type")
        if not model_type:
            raise ValueError("hyperparameter_tune requires 'model_type' in params")

        n_trials = int(params.get("n_trials", 10))
        max_time = int(params.get("max_time_minutes", 0))
        random_state = int(params.get("random_state", 42))
        random.seed(random_state)

        # Very small search spaces for MVP
        search_spaces = {
            "classification_rf": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10],
            },
            "classification_xgboost": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.05, 0.1],
            },
            "classification_logreg": {"C": [0.01, 0.1, 1.0, 10.0]},
            "regression_rf": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
            "regression_linear": {"fit_intercept": [True, False]},
        }

        space = search_spaces.get(model_type, {})

        def sample_params(space):
            out = {}
            for k, choices in space.items():
                out[k] = random.choice(choices)
            return out

        best_score = -np.inf
        best_context = None
        start = time.time()

        for i in range(max(1, n_trials)):
            if max_time and (time.time() - start) > (max_time * 60):
                break

            sampled = sample_params(space) if space else {}

            # Build a lightweight fake stage for the underlying model trainer
            class FakeStage:
                pass

            fake_stage = FakeStage()
            fake_stage.type = params.get("model_type")
            fake_stage.params = {**(params.get("model_params", {}) or {}), **sampled}
            fake_stage.models = []

            handler_cls = get_handler_for_stage(fake_stage.type)
            handler = handler_cls(fake_stage)

            # Use a deep copy of context so failed runs don't mutate the main context
            trial_context = copy.deepcopy(context)
            try:
                trial_result = handler.run(trial_context)
            except Exception as e:
                # Skip failing configurations
                print(f"[hyperparameter_tune] trial {i} failed: {e}")
                continue

            metrics = trial_result.get("metrics") or {}
            # Choose primary metric based on task
            task = params.get("task_type", "classification")
            if task == "regression":
                score = -float(metrics.get("rmse", 0.0))
            else:
                score = float(metrics.get("f1", metrics.get("roc_auc", metrics.get("accuracy", 0.0))))

            if score > best_score:
                best_score = score
                best_context = trial_result

        if best_context is None:
            raise RuntimeError("Hyperparameter tuning found no successful candidate")

        # Winner's artifacts become the run context
        context.update(best_context)
        context["tuning_summary"] = {"best_score": float(best_score)}
        return context
