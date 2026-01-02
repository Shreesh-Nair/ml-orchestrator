# core/handler_registry.py
from __future__ import annotations

from typing import Type

from core.yaml_parser import Stage
from handlers.base import BaseHandler
from handlers.data.csv_loader import CSVLoaderHandler
from handlers.preprocess.tabular_preprocess import TabularPreprocessHandler
from handlers.models.classification.random_forest import RandomForestTrainHandler
from handlers.models.anomaly.isolation_forest import IsolationForestHandler  # NEW
from handlers.models.regression.random_forest import RandomForestRegressionHandler



_HANDLER_REGISTRY: dict[str, Type[BaseHandler]] = {
    "csv_loader": CSVLoaderHandler,
    "tabular_preprocess": TabularPreprocessHandler,
    "classification": RandomForestTrainHandler,
    "regression": RandomForestRegressionHandler,
    "anomaly_isolation_forest": IsolationForestHandler,  # NEW
}


def get_handler_for_stage(stage: Stage) -> Type[BaseHandler]:
    try:
        return _HANDLER_REGISTRY[stage.type]
    except KeyError as e:
        raise ValueError(f"No handler registered for stage type: {stage.type!r}") from e
