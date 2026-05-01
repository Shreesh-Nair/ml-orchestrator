# core/handler_registry.py
"""
Central registry mapping stage type strings (from YAML) to handler classes.
"""
from typing import Dict, Type

from handlers.base import BaseHandler

# Data loaders
from handlers.data.csv_loader import CSVLoaderHandler

# Preprocessing
from handlers.preprocess.tabular_preprocess import TabularPreprocessHandler

# Classification models
from handlers.models.classification.logistic_regression import LogisticRegressionHandler
from handlers.models.classification.random_forest import RandomForestTrainHandler
from handlers.models.classification.gradient_boosting import GradientBoostingHandler

# Regression models
from handlers.models.regression.linear_regression import LinearRegressionHandler
from handlers.models.regression.random_forest import RandomForestRegressionHandler
from handlers.models.regression.regularized_regression import RegularizedRegressionHandler

# Anomaly detection
from handlers.models.anomaly.isolation_forest import IsolationForestHandler
from handlers.models.anomaly.lof import LOFHandler
from handlers.models.anomaly.one_class_svm import OneClassSVMHandler

# Evaluation
from handlers.evaluate.feature_importance import FeatureImportanceHandler
# Hyperparameter tuning
from handlers.models.tuning.hyperparameter_tuner import HyperparameterTunerHandler


_HANDLER_REGISTRY: Dict[str, Type[BaseHandler]] = {
    # Data loaders
    "csv_loader": CSVLoaderHandler,

    # Preprocessing
    "tabular_preprocess": TabularPreprocessHandler,

    # Classification models
    "classification": RandomForestTrainHandler,          # Backward-compatible default
    "classification_rf": RandomForestTrainHandler,
    "classification_logreg": LogisticRegressionHandler,
    "classification_xgboost": GradientBoostingHandler,

    # Regression models
    "regression": RandomForestRegressionHandler,         # Backward-compatible default
    "regression_rf": RandomForestRegressionHandler,
    "regression_linear": LinearRegressionHandler,
    "regression_ridge": RegularizedRegressionHandler,
    "regression_lasso": RegularizedRegressionHandler,
    "regression_elasticnet": RegularizedRegressionHandler,

    # Anomaly detection
    "anomaly_isolation_forest": IsolationForestHandler,
    "anomaly_lof": LOFHandler,
    "anomaly_ocsvm": OneClassSVMHandler,

    # Evaluation
    "feature_importance": FeatureImportanceHandler,
    # Hyperparameter tuning
    "hyperparameter_tune": HyperparameterTunerHandler,
}


def get_handler_for_stage(stage) -> Type[BaseHandler]:
    """
    Returns the handler class for the given stage type.
    """
    stage_type = stage.type if hasattr(stage, "type") else stage

    if stage_type not in _HANDLER_REGISTRY:
        available = ", ".join(_HANDLER_REGISTRY.keys())
        raise KeyError(f"Unknown stage type: '{stage_type}'. Available types: {available}")

    return _HANDLER_REGISTRY[stage_type]


def list_handlers() -> Dict[str, Type[BaseHandler]]:
    """Returns a copy of the entire handler registry."""
    return _HANDLER_REGISTRY.copy()
