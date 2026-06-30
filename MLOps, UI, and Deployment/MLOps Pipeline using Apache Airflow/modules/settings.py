"""Project settings utilities.

Loads YAML config once and provides path helpers used across modules and DAGs.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """Return repository root path."""
    return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load project configuration YAML.

    Args:
        config_path: Relative or absolute path to config file.

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = get_project_root() / config_path

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")

    return data


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a path key from `paths` config block to absolute path."""
    rel = config.get("paths", {}).get(key)
    if rel is None:
        raise KeyError(f"Missing paths.{key} in config")
    path = get_project_root() / rel
    return path


def ensure_parent(path: Path) -> Path:
    """Ensure parent directory exists for path, return same path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_directories(config: dict[str, Any]) -> None:
    """Create all configured output/data parent directories."""
    for _, rel in config.get("paths", {}).items():
        path = get_project_root() / rel
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)


def bootstrap_runtime_env() -> None:
    """Set safe local defaults for Airflow/Matplotlib runtime paths."""
    root = get_project_root()
    airflow_home = Path(os.environ.setdefault("AIRFLOW_HOME", str(root / "airflow_config")))
    os.environ.setdefault("AIRFLOW__CORE__DAGS_FOLDER", str(root / "dags"))
    os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".tmp" / "matplotlib"))
    airflow_home.mkdir(parents=True, exist_ok=True)
    (root / ".tmp" / "matplotlib").mkdir(parents=True, exist_ok=True)


bootstrap_runtime_env()
