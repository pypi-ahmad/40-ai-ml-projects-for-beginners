"""Collect runtime telemetry snapshots."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from llmft.utils.io import write_json


@dataclass(slots=True)
class RuntimeSnapshot:
    """Runtime metric snapshot."""

    timestamp: float
    gpu_util: float | None
    gpu_mem_mb: float | None
    cpu_load: float


class RuntimeMonitor:
    """Collect and persist runtime metrics."""

    def capture(self) -> RuntimeSnapshot:
        """Capture single metrics snapshot."""
        gpu_util, gpu_mem = self._gpu_metrics()
        cpu_load = self._cpu_load()
        return RuntimeSnapshot(timestamp=time.time(), gpu_util=gpu_util, gpu_mem_mb=gpu_mem, cpu_load=cpu_load)

    def write(self, path: str | Path, snapshot: RuntimeSnapshot) -> None:
        """Write snapshot to JSON file."""
        write_json(path, asdict(snapshot))

    def _gpu_metrics(self) -> tuple[float | None, float | None]:
        if shutil.which("nvidia-smi") is None:
            return None, None
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(cmd, text=True).strip().splitlines()[0]
            util, mem = output.split(",")
            return float(util.strip()), float(mem.strip())
        except Exception:  # noqa: BLE001
            return None, None

    def _cpu_load(self) -> float:
        try:
            load1, _, _ = tuple(__import__("os").getloadavg())
            return float(load1)
        except Exception:  # noqa: BLE001
            return 0.0
