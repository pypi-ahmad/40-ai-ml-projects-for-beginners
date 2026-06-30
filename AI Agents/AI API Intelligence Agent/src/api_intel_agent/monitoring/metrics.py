"""Runtime monitoring metrics collection."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import psutil

try:
    from pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName,
        nvmlInit,
    )
except Exception:  # pragma: no cover
    nvmlInit = None


@dataclass
class MetricsSnapshot:
    cpu_percent: float
    memory_percent: float
    gpu_name: str | None
    gpu_vram_used_mb: float | None
    gpu_vram_total_mb: float | None
    cache_hit_rate: float
    api_latency_ms: dict[str, float] = field(default_factory=dict)
    llm_latency_ms: float | None = None
    tokens_per_sec: float | None = None


class MetricsCollector:
    def __init__(self) -> None:
        self.api_latency_ms: dict[str, list[float]] = {}
        self.cache_hits = 0
        self.cache_total = 0

    def record_api_latency(self, provider: str, latency_ms: float) -> None:
        self.api_latency_ms.setdefault(provider, []).append(latency_ms)

    def record_cache(self, hit: bool) -> None:
        self.cache_total += 1
        if hit:
            self.cache_hits += 1

    def snapshot(self, llm_latency_ms: float | None = None, token_count: int | None = None) -> MetricsSnapshot:
        gpu_name = None
        used_mb = None
        total_mb = None

        if nvmlInit:
            try:
                nvmlInit()
                handle = nvmlDeviceGetHandleByIndex(0)
                gpu_name = nvmlDeviceGetName(handle).decode("utf-8")
                mem = nvmlDeviceGetMemoryInfo(handle)
                used_mb = round(mem.used / (1024 * 1024), 2)
                total_mb = round(mem.total / (1024 * 1024), 2)
            except Exception:
                gpu_name = None

        cache_hit_rate = (self.cache_hits / self.cache_total) if self.cache_total else 0.0
        avg_latency = {
            provider: round(sum(values) / len(values), 3)
            for provider, values in self.api_latency_ms.items()
            if values
        }

        tokens_per_sec = None
        if llm_latency_ms and token_count:
            tokens_per_sec = round(token_count / max(llm_latency_ms / 1000.0, 1e-6), 3)

        return MetricsSnapshot(
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=psutil.virtual_memory().percent,
            gpu_name=gpu_name,
            gpu_vram_used_mb=used_mb,
            gpu_vram_total_mb=total_mb,
            cache_hit_rate=round(cache_hit_rate, 3),
            api_latency_ms=avg_latency,
            llm_latency_ms=llm_latency_ms,
            tokens_per_sec=tokens_per_sec,
        )


class ProgressTracker:
    def __init__(self) -> None:
        self._started_at = time.time()

    def elapsed_seconds(self) -> float:
        return round(time.time() - self._started_at, 3)

    def pid(self) -> int:
        return os.getpid()
