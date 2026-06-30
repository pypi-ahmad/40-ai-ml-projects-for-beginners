"""System monitoring helpers (CPU, memory, GPU/VRAM if available)."""

from __future__ import annotations

import subprocess
from typing import Any

import psutil


def _gpu_stats() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return {"available": False}

    rows = []
    for line in output.strip().splitlines():
        name, util, used, total = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "name": name,
                "utilization_gpu_percent": float(util),
                "memory_used_mb": float(used),
                "memory_total_mb": float(total),
            }
        )
    return {"available": True, "gpus": rows}


def system_snapshot() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.3),
        "memory": {
            "total_mb": vm.total / (1024 * 1024),
            "used_mb": vm.used / (1024 * 1024),
            "percent": vm.percent,
        },
        "gpu": _gpu_stats(),
    }
