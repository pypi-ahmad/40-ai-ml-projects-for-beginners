"""Streaming helpers for incremental answer rendering (phase-2 optional)."""

from __future__ import annotations

from collections.abc import Iterator


def stream_text(text: str, chunk_size: int = 32) -> Iterator[str]:
    """Yield fixed-size chunks for streaming UX."""

    for idx in range(0, len(text), chunk_size):
        yield text[idx : idx + chunk_size]
