"""Performance and resource profiling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass(slots=True)
class ResourceSnapshot:
    """Runtime and storage resource snapshot."""

    rss_mb: float
    cpu_percent: float
    disk_usage_mb: float


def directory_size_mb(path: Path) -> float:
    """Compute directory size in MB."""

    if not path.exists():
        return 0.0

    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total / (1024 * 1024)


def capture_resource_snapshot(vector_db_path: Path) -> ResourceSnapshot:
    """Capture current process resource metrics."""

    process = psutil.Process()
    rss_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent(interval=0.1)
    disk_usage_mb = directory_size_mb(vector_db_path)
    return ResourceSnapshot(rss_mb=rss_mb, cpu_percent=cpu_percent, disk_usage_mb=disk_usage_mb)
