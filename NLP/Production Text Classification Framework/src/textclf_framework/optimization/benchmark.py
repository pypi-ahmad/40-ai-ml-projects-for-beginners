"""Inference benchmark helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(slots=True)
class InferenceBenchmarkResult:
    framework: str
    mean_latency_ms: float
    p95_latency_ms: float
    throughput_rps: float
    model_size_mb: float


def _size_mb(path: Path) -> float:
    if not path.exists() or not path.is_file():
        return 0.0
    return float(path.stat().st_size / (1024 * 1024))


def benchmark_predict_fn(
    predict_fn: Callable[[], None],
    repeats: int,
    model_path: str | Path,
    framework: str,
) -> InferenceBenchmarkResult:
    """Benchmark latency and throughput for any callable prediction function."""
    timings_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict_fn()
        timings_ms.append((time.perf_counter() - start) * 1000)

    timings = np.asarray(timings_ms)
    mean_ms = float(np.mean(timings))
    throughput = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return InferenceBenchmarkResult(
        framework=framework,
        mean_latency_ms=mean_ms,
        p95_latency_ms=float(np.percentile(timings, 95)),
        throughput_rps=throughput,
        model_size_mb=_size_mb(Path(model_path)),
    )
