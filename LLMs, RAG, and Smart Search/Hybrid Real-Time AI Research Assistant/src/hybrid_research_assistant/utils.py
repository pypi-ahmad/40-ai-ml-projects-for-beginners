"""General utility helpers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def sha256_text(content: str) -> str:
    """Return sha256 hash for text content."""

    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    """Return sha256 hash for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON payload using deterministic formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def json_load(path: Path) -> dict[str, Any]:
    """Load JSON payload from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows to JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one row to a JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Load a JSONL file into memory."""

    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity for two vectors."""

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
