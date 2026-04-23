from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List


APP_NAME = "ML Orchestrator"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _runtime_roots() -> List[Path]:
    roots: List[Path] = []

    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        roots.append(exe_dir)

        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            roots.append(Path(meipass))
    else:
        roots.append(project_root())

    # Keep order, remove duplicates
    unique: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def find_resource(*parts: str) -> Path | None:
    candidates = [root.joinpath(*parts) for root in _runtime_roots()]
    return _first_existing(candidates)


def get_examples_dir() -> Path:
    found = find_resource("examples")
    if found is not None:
        return found
    return _runtime_roots()[0] / "examples"


def get_data_dir() -> Path:
    found = find_resource("data")
    if found is not None:
        return found
    return _runtime_roots()[0] / "data"


def get_demo_dataset_path() -> Path:
    found = find_resource("data", "titanic.csv")
    if found is None:
        raise FileNotFoundError("Demo dataset not found: data/titanic.csv")
    return found


def get_user_data_dir() -> Path:
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        base = Path(local_app_data)
    else:
        # Fallback for non-Windows environments.
        base = Path.home() / ".local" / "share"

    path = base / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_models_dir() -> Path:
    path = get_user_data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logs_dir() -> Path:
    path = get_user_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_runs_csv_path() -> Path:
    return get_logs_dir() / "runs.csv"


def get_run_summaries_dir() -> Path:
    path = get_logs_dir() / "run_summaries"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_generated_pipelines_dir() -> Path:
    path = get_user_data_dir() / "pipelines" / "generated"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_projects_dir() -> Path:
    path = get_user_data_dir() / "projects"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_exports_dir() -> Path:
    path = get_user_data_dir() / "exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_source_path(path_text: str, *, pipeline_dir: str | None = None) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    search_roots: List[Path] = []
    if pipeline_dir:
        search_roots.append(Path(pipeline_dir))

    search_roots.extend(_runtime_roots())
    search_roots.append(Path.cwd())

    for root in search_roots:
        resolved = root / candidate
        if resolved.exists():
            return resolved

    return candidate
