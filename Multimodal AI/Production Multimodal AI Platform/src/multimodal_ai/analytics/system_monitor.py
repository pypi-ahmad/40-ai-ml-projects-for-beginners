"""System resource telemetry."""

from __future__ import annotations

from typing import Any


def collect_system_stats() -> dict[str, Any]:
    """Collect CPU/GPU telemetry when available."""

    payload: dict[str, Any] = {}

    try:
        import psutil

        payload["cpu_percent"] = psutil.cpu_percent(interval=0.05)
        payload["memory_percent"] = psutil.virtual_memory().percent
    except Exception:  # noqa: BLE001
        payload["cpu_percent"] = None
        payload["memory_percent"] = None

    try:
        import torch

        payload["gpu_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            payload["gpu_name"] = torch.cuda.get_device_name(0)
            payload["vram_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024**2), 2)
            payload["vram_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024**2), 2)
    except Exception:  # noqa: BLE001
        payload.setdefault("gpu_available", False)

    return payload
