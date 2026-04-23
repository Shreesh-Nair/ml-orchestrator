# core/executor.py
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

from core.yaml_parser import parse_pipeline
from core.handler_registry import get_handler_for_stage
from core.run_logger import log_run
from core.run_summary import write_run_summary


class PipelineExecutionError(RuntimeError):
    """Raised when a pipeline stage fails with user-facing context."""


def _error_hint(message: str) -> str:
    lower = message.lower()
    if "not found" in lower and "csv" in lower:
        return "Check the dataset path and confirm the file exists and is readable."
    if "target" in lower:
        return "Verify the selected target column exists and has valid values."
    if "exactly 2" in lower or "binary" in lower:
        return "Use a binary target column or switch to a non-binary workflow."
    if "empty" in lower:
        return "Use a dataset with rows and valid target values."
    return "Review stage parameters and input data, then try again."


def run_pipeline(yaml_path: str) -> Dict[str, Any]:
    resolved_yaml_path = Path(yaml_path).resolve()
    config = parse_pipeline(resolved_yaml_path)
    print(f"[executor] Running pipeline: {config.pipeline_name!r}")
    run_timestamp = datetime.now(timezone.utc)
    context: Dict[str, Any] = {
        "_pipeline_path": str(resolved_yaml_path),
        "_pipeline_dir": str(resolved_yaml_path.parent),
        "_run_timestamp": run_timestamp.isoformat(),
        "_run_id": run_timestamp.strftime("%Y%m%dT%H%M%S%fZ"),
    }

    for stage in config.stages:
        print(f"[executor] Stage: {stage.name!r} (type={stage.type})")
        handler_cls = get_handler_for_stage(stage)
        handler = handler_cls(stage)
        try:
            context = handler.run(context)
        except Exception as exc:
            hint = _error_hint(str(exc))
            raise PipelineExecutionError(
                f"Stage '{stage.name}' ({stage.type}) failed: {exc}\nHint: {hint}"
            ) from exc

    print("[executor] Pipeline complete.")
    if "metrics" in context:
        print(f"[executor] Final metrics: {context['metrics']}")
    if "anomaly_metrics" in context:
        print(f"[executor] Anomaly metrics: {context['anomaly_metrics']}")

    log_run(config.pipeline_name, context)
    write_run_summary(config.pipeline_name, context)

    return context



def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Usage: python -m core.executor <pipeline_yaml_path>")
        raise SystemExit(1)

    yaml_path = argv[0]
    run_pipeline(yaml_path)


if __name__ == "__main__":
    main()
