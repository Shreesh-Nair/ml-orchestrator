# core/executor.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

from core.yaml_parser import parse_pipeline
from core.handler_registry import get_handler_for_stage
from core.run_logger import log_run  # add import

def run_pipeline(yaml_path: str) -> Dict[str, Any]:
    resolved_yaml_path = Path(yaml_path).resolve()
    config = parse_pipeline(resolved_yaml_path)
    print(f"[executor] Running pipeline: {config.pipeline_name!r}")
    context: Dict[str, Any] = {
        "_pipeline_path": str(resolved_yaml_path),
        "_pipeline_dir": str(resolved_yaml_path.parent),
    }

    for stage in config.stages:
        print(f"[executor] Stage: {stage.name!r} (type={stage.type})")
        handler_cls = get_handler_for_stage(stage)
        handler = handler_cls(stage)
        context = handler.run(context)

    print("[executor] Pipeline complete.")
    if "metrics" in context:
        print(f"[executor] Final metrics: {context['metrics']}")
    if "anomaly_metrics" in context:
        print(f"[executor] Anomaly metrics: {context['anomaly_metrics']}")

    # NEW: log this run
    log_run(config.pipeline_name, context)

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
