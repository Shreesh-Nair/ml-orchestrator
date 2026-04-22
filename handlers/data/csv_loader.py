# handlers/data/csv_loader.py
from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from core.paths import resolve_source_path
from handlers.base import BaseHandler


class CSVLoaderHandler(BaseHandler):
    """
    Loads a CSV file into a pandas DataFrame and stores it in context["df"].
    Expects:
      stage.params["source"]: path to CSV
      stage.params["target_column"]: name of target column (string)
    """

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        source = self.stage.params.get("source")
        target_column = self.stage.params.get("target_column")

        if not isinstance(source, str) or not source:
            raise ValueError(f"CSVLoaderHandler: 'source' must be a non-empty string, got {source!r}")
        if not isinstance(target_column, str) or not target_column:
            raise ValueError(
                f"CSVLoaderHandler: 'target_column' must be a non-empty string, got {target_column!r}"
            )

        path = resolve_source_path(
            source,
            pipeline_dir=context.get("_pipeline_dir"),
        )
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path)
        context["df"] = df
        context["target_column"] = target_column

        print(f"[csv_loader] Loaded {df.shape[0]} rows, {df.shape[1]} columns from {path}")
        return context
