"""Shared utility helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize whitespace and strip control chars."""
    compact = WHITESPACE_RE.sub(" ", text.replace("\x00", " ")).strip()
    return compact


def hash_text(text: str) -> str:
    """Stable sha256 hash for deduplication and caching."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_filename(value: str) -> str:
    """Filesystem-safe name for generated files."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return Path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def dump_json(path: str | Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    """Write JSON with stable formatting."""
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
