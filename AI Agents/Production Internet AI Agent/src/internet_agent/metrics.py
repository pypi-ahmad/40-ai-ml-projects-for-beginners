"""Simple in-memory metrics store."""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(slots=True)
class MetricsStore:
    """Thread-safe metrics collector for counters and latency samples."""

    _counters: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _latencies_ms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, key: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[key] += value

    def observe_ms(self, key: str, value: float) -> None:
        with self._lock:
            self._latencies_ms[key].append(value)

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            latency_summary = {
                key: {
                    "count": float(len(values)),
                    "avg_ms": (sum(values) / len(values)) if values else 0.0,
                    "max_ms": max(values) if values else 0.0,
                    "min_ms": min(values) if values else 0.0,
                }
                for key, values in self._latencies_ms.items()
            }
            counters = dict(self._counters)
        return {"counters": counters, "latencies": latency_summary}


METRICS = MetricsStore()
