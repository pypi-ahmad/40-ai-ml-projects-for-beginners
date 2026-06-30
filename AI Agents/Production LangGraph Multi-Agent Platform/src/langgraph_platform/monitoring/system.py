"""System metrics collectors for CPU/GPU and workflow timing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import psutil


@dataclass(slots=True)
class NodeTiming:
    """Node timing record."""

    node: str
    started: float


class SystemMonitor:
    """Collect runtime and hardware metrics."""

    def __init__(self, enable_gpu_metrics: bool = True) -> None:
        self.enable_gpu_metrics = enable_gpu_metrics
        self._timers: dict[str, NodeTiming] = {}

    def start_node(self, node: str) -> None:
        self._timers[node] = NodeTiming(node=node, started=time.perf_counter())

    def stop_node(self, node: str) -> float:
        timer = self._timers.pop(node, None)
        if timer is None:
            return 0.0
        return (time.perf_counter() - timer.started) * 1000.0

    def capture(self) -> dict[str, Any]:
        """Capture CPU/memory/GPU snapshot."""

        metrics: dict[str, Any] = {
            "cpu_percent": psutil.cpu_percent(interval=0.05),
            "memory_percent": psutil.virtual_memory().percent,
            "process_rss_mb": psutil.Process().memory_info().rss / (1024 * 1024),
        }

        if self.enable_gpu_metrics:
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics["gpu_util_percent"] = util.gpu
                metrics["vram_used_mb"] = mem.used / (1024 * 1024)
                metrics["vram_total_mb"] = mem.total / (1024 * 1024)
            except Exception:
                metrics["gpu_util_percent"] = None
                metrics["vram_used_mb"] = None
                metrics["vram_total_mb"] = None

        return metrics
