from __future__ import annotations

from typing import Any, Dict
import copy
import pickle
import random
import time

import numpy as np

from handlers.base import BaseHandler


class HyperparameterTunerHandler(BaseHandler):
    """Hyperparameter tuner using randomized search over predefined spaces.
    
    Features:
    - Randomized search over sensible parameter ranges for each model type
    - Baseline model comparison (default params vs. tuned)
    - Early stopping (stop if no improvement in N trials)
    - Tracks training time and provides a comparison report
    """

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from core.handler_registry import get_handler_for_stage
        
        params = self.stage.params or {}
        model_type = params.get("model_type")
        if not model_type:
            raise ValueError("hyperparameter_tune requires 'model_type' in params")

        n_trials = int(params.get("n_trials", 10))
        max_time = int(params.get("max_time_minutes", 0))
        random_state = int(params.get("random_state", 42))
        random.seed(random_state)
        np.random.seed(random_state)

        # Expanded search spaces for better tuning
        search_spaces = {
            "classification_rf": {
                "n_estimators": [50, 100, 150, 200, 300],
                "max_depth": [5, 7, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "classification_xgboost": {
                "n_estimators": [50, 100, 150, 200],
                "max_depth": [3, 5, 7, 10],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 0.9, 1.0],
            },
            "classification_logreg": {
                "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                "max_iter": [100, 200, 500],
            },
            "regression_rf": {
                "n_estimators": [50, 100, 150, 200],
                "max_depth": [5, 7, 10, 15, None],
                "min_samples_split": [2, 5, 10],
            },
            "regression_linear": {
                "fit_intercept": [True, False],
            },
            "regression_ridge": {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            },
            "regression_lasso": {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                "max_iter": [1000, 2000],
            },
        }

        space = search_spaces.get(model_type, {})

        def sample_params(space):
            out = {}
            for k, choices in space.items():
                out[k] = random.choice(choices)
            return out

        # Step 1: Train baseline model (default params)
        baseline_context = self._train_model(context, model_type, {}, params, random_state)
        baseline_metrics = baseline_context.get("metrics", {})
        baseline_score = self._extract_score(baseline_metrics, params.get("task_type", "classification"))

        # Step 2: Run hyperparameter search with early stopping
        best_score = baseline_score
        best_context = baseline_context
        best_params = {}
        trials_without_improvement = 0
        early_stop_patience = max(2, n_trials // 5)  # Stop if no improvement in N trials
        start = time.time()
        trial_num = 0

        while trial_num < n_trials:
            if max_time and (time.time() - start) > (max_time * 60):
                break

            # Early stopping
            if trials_without_improvement >= early_stop_patience:
                print(f"[hyperparameter_tune] Early stopping: no improvement in {early_stop_patience} trials")
                break

            sampled = sample_params(space) if space else {}
            trial_context = self._train_model(context, model_type, sampled, params, random_state)
            
            if trial_context is None:
                trials_without_improvement += 1
                trial_num += 1
                continue

            metrics = trial_context.get("metrics") or {}
            score = self._extract_score(metrics, params.get("task_type", "classification"))

            if score > best_score:
                best_score = score
                best_context = trial_context
                best_params = sampled
                trials_without_improvement = 0
                print(f"[hyperparameter_tune] Trial {trial_num}: NEW BEST score={score:.4f}, params={sampled}")
            else:
                trials_without_improvement += 1

            trial_num += 1

        # Step 3: Prepare comparison report
        elapsed_time = time.time() - start
        baseline_model_size_bytes = self._estimate_model_size_bytes(baseline_context.get("model"))
        best_model_size_bytes = self._estimate_model_size_bytes(best_context.get("model"))
        size_delta_bytes = None
        size_change_pct = None
        if baseline_model_size_bytes is not None and best_model_size_bytes is not None:
            size_delta_bytes = best_model_size_bytes - baseline_model_size_bytes
            if baseline_model_size_bytes != 0:
                size_change_pct = float(size_delta_bytes / baseline_model_size_bytes * 100)

        recommendation = self._build_recommendation(
            baseline_score=baseline_score,
            best_score=best_score,
            size_change_pct=size_change_pct,
            elapsed_seconds=elapsed_time,
        )

        comparison_report = {
            "baseline_score": float(baseline_score),
            "best_score": float(best_score),
            "improvement_pct": float((best_score - baseline_score) / abs(baseline_score) * 100) if baseline_score != 0 else 0.0,
            "best_params": best_params,
            "trials_run": trial_num,
            "elapsed_seconds": float(elapsed_time),
            "baseline_metrics": baseline_metrics,
            "best_metrics": best_context.get("metrics", {}),
            "baseline_model_size_bytes": baseline_model_size_bytes,
            "best_model_size_bytes": best_model_size_bytes,
            "size_delta_bytes": size_delta_bytes,
            "size_change_pct": size_change_pct,
            "recommendation": recommendation,
        }

        # Update context with best model
        context.update(best_context)
        context["tuning_summary"] = comparison_report
        context["baseline_model"] = baseline_context.get("model")
        
        return context

    def _train_model(
        self, context: Dict[str, Any], model_type: str, 
        sampled_params: Dict[str, Any], stage_params: Dict[str, Any],
        random_state: int
    ) -> Dict[str, Any] | None:
        """Train a model with given parameters. Returns context or None on failure."""
        from core.handler_registry import get_handler_for_stage

        class FakeStage:
            pass

        fake_stage = FakeStage()
        fake_stage.type = model_type
        fake_stage.params = {**(stage_params.get("model_params", {}) or {}), **sampled_params}
        fake_stage.models = []

        try:
            handler_cls = get_handler_for_stage(fake_stage.type)
            handler = handler_cls(fake_stage)
            trial_context = copy.deepcopy(context)
            trial_result = handler.run(trial_context)
            return trial_result
        except Exception as e:
            print(f"[hyperparameter_tune] Model training failed: {e}")
            return None

    def _extract_score(self, metrics: Dict[str, Any], task_type: str) -> float:
        """Extract the primary metric for scoring based on task type."""
        if task_type == "regression":
            return -float(metrics.get("rmse", 0.0))
        elif task_type == "anomaly":
            return float(metrics.get("roc_auc", metrics.get("accuracy", 0.0)))
        else:  # classification
            return float(metrics.get("f1", metrics.get("roc_auc", metrics.get("accuracy", 0.0))))

    def _estimate_model_size_bytes(self, model: Any) -> int | None:
        """Estimate serialized model size in bytes."""
        if model is None:
            return None

        try:
            return len(pickle.dumps(model))
        except Exception as exc:
            print(f"[hyperparameter_tune] Model size estimation failed: {exc}")
            return None

    def _build_recommendation(
        self,
        baseline_score: float,
        best_score: float,
        size_change_pct: float | None,
        elapsed_seconds: float,
    ) -> str:
        """Build a simple production recommendation from tuning results."""
        improvement_pct = 0.0
        if baseline_score != 0:
            improvement_pct = float((best_score - baseline_score) / abs(baseline_score) * 100)

        if best_score <= baseline_score:
            return "Keep the baseline model; tuning did not improve the selected metric."

        if improvement_pct >= 5.0 and (size_change_pct is None or size_change_pct <= 25.0) and elapsed_seconds <= 600:
            return "Recommended for production after standard validation checks."

        return "Promising candidate; review metric gain, runtime, and model size before production."
