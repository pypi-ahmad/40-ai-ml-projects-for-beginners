"""System monitoring metrics collection."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import psutil

try:
    import pynvml
except Exception:  # pragma: no cover
    pynvml = None


class MonitoringService:
    """Collects CPU/GPU/runtime operational metrics."""

    def collect(self) -> dict[str, Any]:
        cpu = psutil.cpu_percent(interval=0.05)
        memory = psutil.virtual_memory()
        metrics: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu_percent": cpu,
            "ram_percent": memory.percent,
            "ram_used_mb": round(memory.used / (1024 * 1024), 2),
            "gpu_available": False,
            "gpu_name": None,
            "gpu_memory_total_mb": None,
            "gpu_memory_used_mb": None,
        }

        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                metrics.update(
                    {
                        "gpu_available": True,
                        "gpu_name": name.decode() if isinstance(name, bytes) else str(name),
                        "gpu_memory_total_mb": round(info.total / (1024 * 1024), 2),
                        "gpu_memory_used_mb": round(info.used / (1024 * 1024), 2),
                    }
                )
                pynvml.nvmlShutdown()
            except Exception:
                pass

        return metrics
