"""Runtime metrics collection utilities."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MetricEvent:
    """Single timing event."""

    name: str
    value_ms: float
    timestamp: float


class MetricsCollector:
    """In-memory rolling metrics collector."""

    def __init__(self, max_events: int = 2000) -> None:
        self._events: deque[MetricEvent] = deque(maxlen=max_events)

    def record(self, name: str, value_ms: float) -> None:
        """Record latency event."""

        self._events.append(MetricEvent(name=name, value_ms=value_ms, timestamp=time.time()))

    def summary(self) -> dict[str, dict[str, float]]:
        """Return aggregated metric summary."""

        grouped: dict[str, list[float]] = {}
        for event in self._events:
            grouped.setdefault(event.name, []).append(event.value_ms)

        result: dict[str, dict[str, float]] = {}
        for name, values in grouped.items():
            avg = sum(values) / len(values)
            result[name] = {
                "count": float(len(values)),
                "avg_ms": avg,
                "max_ms": max(values),
                "min_ms": min(values),
            }
        return result

    def raw(self) -> list[dict[str, Any]]:
        """Return raw metric events."""

        return [
            {
                "name": event.name,
                "value_ms": event.value_ms,
                "timestamp": event.timestamp,
            }
            for event in self._events
        ]
