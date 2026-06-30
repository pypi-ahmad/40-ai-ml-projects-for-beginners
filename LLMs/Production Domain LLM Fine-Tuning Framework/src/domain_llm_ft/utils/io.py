"""I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    """Read YAML file into dict."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON file with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
