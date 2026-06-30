"""Performance instrumentation helpers."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class PerfResult:
    """Performance measurement output."""

    label: str
    elapsed_ms: float
    peak_memory_mb: float
    metadata: dict[str, Any]


def measure(label: str, fn: Callable[[], Any], metadata: dict[str, Any] | None = None) -> tuple[Any, PerfResult]:
    """Measure runtime and memory for callable."""
    tracemalloc.start()
    start = time.perf_counter()
    output = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return output, PerfResult(
        label=label,
        elapsed_ms=elapsed_ms,
        peak_memory_mb=peak / (1024 * 1024),
        metadata=metadata or {},
    )
