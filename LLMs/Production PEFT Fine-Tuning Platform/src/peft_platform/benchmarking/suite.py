"""Benchmarking utilities."""

from __future__ import annotations

import statistics
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable


@dataclass(slots=True)
class BenchmarkResult:
    latency_ms_avg: float
    latency_ms_p95: float
    throughput_req_per_sec: float
    peak_memory_mb: float


def benchmark_callable(fn: Callable[[], str], runs: int = 10) -> BenchmarkResult:
    latencies: list[float] = []
    tracemalloc.start()
    start = time.perf_counter()

    for _ in range(runs):
        run_start = time.perf_counter()
        _ = fn()
        latencies.append((time.perf_counter() - run_start) * 1000)

    total = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    sorted_lats = sorted(latencies)
    p95_idx = max(int(0.95 * len(sorted_lats)) - 1, 0)

    return BenchmarkResult(
        latency_ms_avg=statistics.mean(latencies),
        latency_ms_p95=sorted_lats[p95_idx],
        throughput_req_per_sec=runs / total if total > 0 else 0.0,
        peak_memory_mb=peak / (1024 * 1024),
    )
