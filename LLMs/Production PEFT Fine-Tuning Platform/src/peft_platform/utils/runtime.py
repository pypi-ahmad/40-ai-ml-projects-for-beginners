"""Runtime and hardware helpers."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeInfo:
    python_version: str
    platform: str
    cuda_available: bool
    device: str


@dataclass(slots=True)
class GPUInfo:
    name: str
    memory_total_mb: int
    driver_version: str


def detect_runtime() -> RuntimeInfo:
    """Detect runtime basics and torch/cuda availability."""
    cuda_available = False
    device = "cpu"
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device = "cuda"
    except Exception:
        cuda_available = False

    return RuntimeInfo(
        python_version=platform.python_version(),
        platform=platform.platform(),
        cuda_available=cuda_available,
        device=device,
    )


def query_gpu() -> GPUInfo | None:
    """Read GPU info from nvidia-smi when available."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None

    first = result.stdout.strip().splitlines()[0]
    name, mem, driver = [part.strip() for part in first.split(",")]
    return GPUInfo(name=name, memory_total_mb=int(mem), driver_version=driver)
