"""In-memory request and latency metrics store."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
import statistics
import time


@dataclass
class MetricsStore:
    """Thread-safe process-local metrics for API observability."""

    started_at: float = field(default_factory=time.time)
    request_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    status_counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    latencies_ms: list[float] = field(default_factory=list)
    route_latencies: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    lock: Lock = field(default_factory=Lock)

    def record(self, route: str, status_code: int, latency_ms: float) -> None:
        """Record request telemetry for one completed request."""
        with self.lock:
            self.request_counts[route] += 1
            self.status_counts[status_code] += 1
            self.latencies_ms.append(latency_ms)
            self.route_latencies[route].append(latency_ms)

    def snapshot(self) -> dict[str, object]:
        """Return current metrics state as JSON-serializable object."""
        with self.lock:
            latencies = list(self.latencies_ms)
            route_stats: dict[str, dict[str, float]] = {}
            for route, values in self.route_latencies.items():
                if not values:
                    continue
                route_stats[route] = {
                    "count": len(values),
                    "avg_ms": round(sum(values) / len(values), 3),
                    "p95_ms": round(_percentile(values, 95), 3),
                }

            return {
                "uptime_seconds": round(time.time() - self.started_at, 3),
                "request_counts": dict(self.request_counts),
                "status_counts": dict(self.status_counts),
                "latency_ms": {
                    "count": len(latencies),
                    "avg": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
                    "p50": round(statistics.median(latencies), 3) if latencies else 0.0,
                    "p95": round(_percentile(latencies, 95), 3) if latencies else 0.0,
                },
                "route_latency_ms": route_stats,
            }


def _percentile(values: list[float], percentile: int) -> float:
    """Compute percentile using nearest-rank interpolation."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
