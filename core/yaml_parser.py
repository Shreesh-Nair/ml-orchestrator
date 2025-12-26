# core/yaml_parser.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # comes from pyyaml

@dataclass
class Stage:
    name: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    models: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"Stage name must be a non-empty string, got: {self.name!r}")
        if not isinstance(self.type, str) or not self.type.strip():
            raise ValueError(f"Stage type must be a non-empty string, got: {self.type!r}")
        if not isinstance(self.params, dict):
            raise ValueError(f"Stage params must be a dict for stage {self.name!r}")
        if self.models is not None and not isinstance(self.models, list):
            raise ValueError(f"Stage models must be a list for stage {self.name!r}")

@dataclass
class PipelineConfig:
    pipeline_name: str
    stages: List[Stage]

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline_name, str) or not self.pipeline_name.strip():
            raise ValueError(f"pipeline_name must be a non-empty string, got: {self.pipeline_name!r}")
        if not isinstance(self.stages, list) or not self.stages:
            raise ValueError("stages must be a non-empty list of Stage objects")

def load_yaml(file_path: str | Path) -> Dict[str, Any]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise OSError(f"Failed to read YAML file {path}: {e}") from e

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {path}: {e}") from e

    if data is None:
        raise ValueError(f"YAML file {path} is empty")
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML content in {path} must be a mapping (dict)")

    return data

def parse_pipeline(file_path: str | Path) -> PipelineConfig:
    config_dict = load_yaml(file_path)

    # pipeline_name
    pipeline_name = config_dict.get("pipeline_name")
    if not isinstance(pipeline_name, str) or not pipeline_name.strip():
        raise ValueError("pipeline_name is required and must be a non-empty string")

    # stages
    stages_raw = config_dict.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise ValueError("stages is required and must be a non-empty list")

    stages: List[Stage] = []

    for idx, stage_dict in enumerate(stages_raw):
        if not isinstance(stage_dict, dict):
            raise ValueError(f"Stage at index {idx} must be a mapping (dict)")

        name = stage_dict.get("name")
        type_ = stage_dict.get("type")

        if name is None:
            raise ValueError(f"Stage at index {idx} is missing 'name'")
        if type_ is None:
            raise ValueError(f"Stage {name!r} is missing 'type'")

        params = stage_dict.get("params", {})
        models = stage_dict.get("models")

        stage = Stage(
            name=name,
            type=type_,
            params=params if params is not None else {},
            models=models,
        )
        stages.append(stage)

    return PipelineConfig(pipeline_name=pipeline_name, stages=stages)
