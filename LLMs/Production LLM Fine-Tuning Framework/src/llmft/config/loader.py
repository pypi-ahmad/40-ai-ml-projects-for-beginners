"""YAML project config loader."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .schemas import ProjectConfig


def _apply_updates(instance: Any, payload: dict[str, Any]) -> None:
    """Apply nested dictionary updates onto a dataclass instance."""
    for item in fields(instance):
        if item.name not in payload:
            continue
        value = payload[item.name]
        current = getattr(instance, item.name)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_updates(current, value)
            continue
        setattr(instance, item.name, value)


def load_project_config(path: str | Path) -> ProjectConfig:
    """Load project config from YAML file.

    Args:
        path: YAML path.

    Returns:
        Parsed ``ProjectConfig``.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = ProjectConfig()
    _apply_updates(config, payload)
    return config
