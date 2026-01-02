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
from handlers.models.classification.random_forest import RandomForestTrainHandler
from handlers.models.classification.logistic_regression import LogisticRegressionHandler

# Regression models
from handlers.models.regression.random_forest import RandomForestRegressionHandler
from handlers.models.regression.linear_regression import LinearRegressionHandler

# Anomaly detection
from handlers.models.anomaly.isolation_forest import IsolationForestHandler


_HANDLER_REGISTRY: Dict[str, Type[BaseHandler]] = {
    # ---------- Data Loaders ----------
    "csv_loader": CSVLoaderHandler,
    
    # ---------- Preprocessing ----------
    "tabular_preprocess": TabularPreprocessHandler,
    
    # ---------- Classification Models ----------
    "classification": RandomForestTrainHandler,              # Default (for backward compatibility)
    "classification_rf": RandomForestTrainHandler,           # Explicit RandomForest
    "classification_logreg": LogisticRegressionHandler,      # ✅ NEW: Logistic Regression
    
    # ---------- Regression Models ----------
    "regression": RandomForestRegressionHandler,             # Default (for backward compatibility)
    "regression_rf": RandomForestRegressionHandler,          # Explicit RandomForest
    "regression_linear": LinearRegressionHandler,            # ✅ NEW: Linear Regression
    
    # ---------- Anomaly Detection ----------
    "anomaly_isolation_forest": IsolationForestHandler,
}


def get_handler_for_stage(stage) -> Type[BaseHandler]:  # Renamed from get_handler
    """
    Returns the handler class for the given stage type.
    """
    # Handles both 'Stage' object (stage.type) or raw string
    stage_type = stage.type if hasattr(stage, "type") else stage
    
    if stage_type not in _HANDLER_REGISTRY:
        available = ", ".join(_HANDLER_REGISTRY.keys())
        raise KeyError(
            f"Unknown stage type: '{stage_type}'. Available types: {available}"
        )
    return _HANDLER_REGISTRY[stage_type]


def list_handlers() -> Dict[str, Type[BaseHandler]]:
    """Returns a copy of the entire handler registry."""
    return _HANDLER_REGISTRY.copy()
