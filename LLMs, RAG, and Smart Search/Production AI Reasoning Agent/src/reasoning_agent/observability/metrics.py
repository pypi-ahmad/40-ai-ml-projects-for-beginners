"""Runtime metric collector."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class MetricCollector:
    """In-memory metric aggregation."""

    counters: Counter[str] = field(default_factory=Counter)
    totals: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))

    def inc(self, key: str, amount: int = 1) -> None:
        """Increment counter metric."""

        self.counters[key] += amount

    def add(self, key: str, value: float) -> None:
        """Add value to aggregate total metric."""

        self.totals[key] += value

    def snapshot(self) -> dict[str, float]:
        """Return flat metric snapshot."""

        data: dict[str, float] = {k: float(v) for k, v in self.counters.items()}
        data.update({k: float(v) for k, v in self.totals.items()})
        return data
