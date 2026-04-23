from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from core.paths import get_run_summaries_dir


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(v) for v in value]
    return str(value)


def write_run_summary(pipeline_name: str, context: Dict[str, Any]) -> Path:
    timestamp = datetime.now(timezone.utc)
    run_id = context.get("_run_id") or timestamp.strftime("%Y%m%dT%H%M%S%fZ")

    summary = {
        "run_id": run_id,
        "timestamp": timestamp.isoformat(),
        "pipeline_name": pipeline_name,
        "pipeline_path": context.get("_pipeline_path"),
        "random_seed": context.get("_random_seed"),
        "metrics": _safe_json_value(context.get("metrics") or {}),
        "anomaly_metrics": _safe_json_value(context.get("anomaly_metrics") or {}),
        "feature_columns": _safe_json_value(context.get("feature_columns") or []),
        "target_column": context.get("target_column"),
        "target_class_counts": _safe_json_value(context.get("target_class_counts") or {}),
    }

    out_dir = get_run_summaries_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}_{pipeline_name}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path
