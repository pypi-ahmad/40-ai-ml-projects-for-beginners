"""System monitoring helpers."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass(slots=True)
class SystemMetrics:
    cpu_percent: float
    memory_mb: float
    gpu_name: str | None
    gpu_memory_mb: int | None


def collect_metrics() -> SystemMetrics:
    load_avg = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    page_size = os.sysconf("SC_PAGE_SIZE")
    phys_pages = os.sysconf("SC_PHYS_PAGES")
    memory_mb = (page_size * phys_pages) / (1024 * 1024)

    gpu_name: str | None = None
    gpu_memory: int | None = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        first = result.stdout.strip().splitlines()[0]
        name, mem = [part.strip() for part in first.split(",")]
        gpu_name = name
        gpu_memory = int(mem)
    except Exception:
        pass

    return SystemMetrics(
        cpu_percent=load_avg,
        memory_mb=memory_mb,
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_memory,
    )
