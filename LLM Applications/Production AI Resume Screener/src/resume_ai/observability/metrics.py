"""Runtime metric collectors for latency and system utilization."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import psutil
from sqlalchemy.orm import Session

from resume_ai.db import models


@dataclass
class MetricsCollector:
    """Collect application and system metrics."""

    sink: Callable[[str, float, dict | None], None] | None = None
    buffer: list[tuple[str, float, dict | None]] = field(default_factory=list)

    def record(self, name: str, value: float, dimensions: dict | None = None) -> None:
        if self.sink:
            self.sink(name, value, dimensions)
        else:
            self.buffer.append((name, value, dimensions))

    @contextmanager
    def timed(self, metric_name: str, dimensions: dict | None = None) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.record(metric_name, elapsed, dimensions)

    def sample_system(self) -> dict[str, float]:
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
        }


def persist_metric(session: Session, name: str, value: float, dimensions: dict | None = None) -> None:
    session.add(
        models.SystemMetric(
            metric_name=name,
            metric_value=value,
            dimension_json=dimensions,
        )
    )
    session.flush()
