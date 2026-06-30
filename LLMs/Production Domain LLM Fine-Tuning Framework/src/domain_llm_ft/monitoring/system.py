"""Runtime monitoring utilities."""

from __future__ import annotations

from dataclasses import dataclass

import psutil
import torch


@dataclass
class SystemSnapshot:
    cpu_percent: float
    ram_percent: float
    gpu_memory_mb: float


def take_snapshot() -> SystemSnapshot:
    """Collect CPU/RAM/GPU utilization snapshot."""
    gpu_memory = 0.0
    if torch.cuda.is_available():
        gpu_memory = float(torch.cuda.memory_allocated() / 1024 / 1024)
    return SystemSnapshot(
        cpu_percent=float(psutil.cpu_percent(interval=None)),
        ram_percent=float(psutil.virtual_memory().percent),
        gpu_memory_mb=gpu_memory,
    )
