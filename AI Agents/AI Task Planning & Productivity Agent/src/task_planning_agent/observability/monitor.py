"""Runtime monitoring for CPU/GPU and process metrics."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import psutil


@dataclass(slots=True)
class RuntimeMetrics:
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    gpu_name: str | None
    gpu_vram_total_mb: float | None


class RuntimeMonitor:
    """Collect runtime system metrics."""

    def collect(self) -> RuntimeMetrics:
        process = psutil.Process()
        mem_info = process.memory_info()
        gpu_name: str | None = None
        gpu_vram: float | None = None
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            line = result.stdout.strip().splitlines()[0]
            name, total = [chunk.strip() for chunk in line.split(",")]
            gpu_name = name
            gpu_vram = float(total)
        except Exception:
            gpu_name = None
            gpu_vram = None

        return RuntimeMetrics(
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=psutil.virtual_memory().percent,
            memory_mb=mem_info.rss / (1024 * 1024),
            gpu_name=gpu_name,
            gpu_vram_total_mb=gpu_vram,
        )
