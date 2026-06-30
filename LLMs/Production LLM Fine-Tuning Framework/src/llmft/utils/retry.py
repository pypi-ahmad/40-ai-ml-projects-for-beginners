"""Retry helper for fragile operations."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def with_retry(func: Callable[[], T], retries: int = 3, delay_seconds: float = 0.5) -> T:
    """Execute callable with basic retry loop."""
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(delay_seconds)
    if last_error is None:
        raise RuntimeError("Retry failed without captured error")
    raise last_error
