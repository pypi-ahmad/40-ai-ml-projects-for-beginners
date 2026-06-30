"""Inference monitoring primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from time import time


@dataclass
class InferenceMetrics:
    request_count: int = 0
    error_count: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    input_lengths: list[int] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    started_at: float = field(default_factory=time)

    def record_success(self, latency_ms: float, input_len: int, confidence: float) -> None:
        self.request_count += 1
        self.latencies_ms.append(latency_ms)
        self.input_lengths.append(input_len)
        self.confidences.append(confidence)

    def record_error(self) -> None:
        self.request_count += 1
        self.error_count += 1

    def snapshot(self) -> dict[str, float]:
        uptime = max(time() - self.started_at, 1e-6)
        return {
            "request_count": float(self.request_count),
            "error_count": float(self.error_count),
            "error_rate": float(self.error_count / self.request_count) if self.request_count else 0.0,
            "mean_latency_ms": float(mean(self.latencies_ms)) if self.latencies_ms else 0.0,
            "p95_latency_ms": float(sorted(self.latencies_ms)[int(0.95 * (len(self.latencies_ms) - 1))])
            if self.latencies_ms
            else 0.0,
            "throughput_rps": float(self.request_count / uptime),
            "avg_input_length": float(mean(self.input_lengths)) if self.input_lengths else 0.0,
            "avg_confidence": float(mean(self.confidences)) if self.confidences else 0.0,
            "uptime_seconds": float(uptime),
        }
