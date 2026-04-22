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

# Regression models
from handlers.models.regression.linear_regression import LinearRegressionHandler
from handlers.models.regression.random_forest import RandomForestRegressionHandler

# Anomaly detection
from handlers.models.anomaly.isolation_forest import IsolationForestHandler


_HANDLER_REGISTRY: Dict[str, Type[BaseHandler]] = {
    # Data loaders
    "csv_loader": CSVLoaderHandler,

    # Preprocessing
    "tabular_preprocess": TabularPreprocessHandler,

    # Classification models
    "classification": RandomForestTrainHandler,          # Backward-compatible default
    "classification_rf": RandomForestTrainHandler,
    "classification_logreg": LogisticRegressionHandler,

    # Regression models
    "regression": RandomForestRegressionHandler,         # Backward-compatible default
    "regression_rf": RandomForestRegressionHandler,
    "regression_linear": LinearRegressionHandler,

    # Anomaly detection
    "anomaly_isolation_forest": IsolationForestHandler,
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
