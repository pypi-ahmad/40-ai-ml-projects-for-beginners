"""Filesystem helper functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON payload with deterministic formatting."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
