"""Shared adapter helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def deterministic_vector(seed_text: str, dim: int = 512) -> list[float]:
    """Generate deterministic pseudo embedding.

    Useful fallback when local model unavailable.
    """

    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    values: list[float] = []
    while len(values) < dim:
        for byte in digest:
            values.append((byte / 255.0) * 2.0 - 1.0)
            if len(values) == dim:
                break
        digest = hashlib.sha256(digest).digest()
    return values


def image_fingerprint(path: str) -> str:
    """Create deterministic identity string from image file."""

    file_path = Path(path)
    if not file_path.exists():
        return f"missing:{path}"
    payload = file_path.read_bytes()
    return hashlib.sha256(payload).hexdigest()
