"""In-memory metric aggregator."""

from __future__ import annotations

from collections import Counter


class MetricsStore:
    """Collects counters and latency values per run."""

    def __init__(self) -> None:
        self.counters: Counter[str] = Counter()
        self.latency_ms: dict[str, list[float]] = {}

    def inc(self, key: str, value: int = 1) -> None:
        self.counters[key] += value

    def observe_ms(self, key: str, value: float) -> None:
        self.latency_ms.setdefault(key, []).append(value)

    def snapshot(self) -> dict[str, object]:
        lat = {
            name: {
                "count": len(values),
                "avg_ms": sum(values) / len(values) if values else 0.0,
                "max_ms": max(values) if values else 0.0,
            }
            for name, values in self.latency_ms.items()
        }
        return {"counters": dict(self.counters), "latency": lat}
