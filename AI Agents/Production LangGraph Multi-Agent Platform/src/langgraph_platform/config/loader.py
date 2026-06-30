"""Hydra-backed configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from langgraph_platform.config.settings import AppConfig, apply_env_overrides


def _merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_dir: str | Path = "configs") -> AppConfig:
    """Load layered YAML config into typed `AppConfig`."""

    config_root = Path(config_dir)
    base_path = config_root / "config.yaml"
    if not base_path.exists():
        return AppConfig()

    with base_path.open("r", encoding="utf-8") as file:
        merged: dict[str, Any] = yaml.safe_load(file) or {}

    for child in config_root.glob("**/*.yaml"):
        if child == base_path:
            continue
        with child.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        merged = _merge_dict(merged, data)

    config = AppConfig.model_validate(merged)
    return apply_env_overrides(config)
