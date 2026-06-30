from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import psutil

from config.settings import MonitoringConfig
from memory.service import MemoryService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeSnapshot:
    cpu_percent: float
    memory_percent: float
    memory_used: float
    active_threads: int


class MetricsCollector:
    def __init__(self, config: MonitoringConfig, memory: MemoryService) -> None:
        self.config = config
        self.memory = memory
        self._task: asyncio.Task[None] | None = None
        self._running = False

    def snapshot(self) -> RuntimeSnapshot:
        proc = psutil.Process()
        mem = psutil.virtual_memory()
        return RuntimeSnapshot(
            cpu_percent=psutil.cpu_percent(interval=0.05),
            memory_percent=mem.percent,
            memory_used=float(mem.used),
            active_threads=proc.num_threads(),
        )

    def record_tool_latency(self, tool_name: str, latency_ms: int) -> None:
        self.memory.log_metric("tool_latency_ms", float(latency_ms), {"tool": tool_name})

    def record_mcp_latency(self, runtime: str, latency_ms: int) -> None:
        self.memory.log_metric("mcp_latency_ms", float(latency_ms), {"runtime": runtime})

    def record_llm_latency(self, model: str, latency_ms: int) -> None:
        self.memory.log_metric("llm_latency_ms", float(latency_ms), {"model": model})

    def record_error(self, category: str) -> None:
        self.memory.log_metric("error_count", 1.0, {"category": category})

    async def _loop(self) -> None:
        while self._running:
            snapshot = self.snapshot()
            self.memory.log_metric("cpu_percent", snapshot.cpu_percent, {})
            self.memory.log_metric("memory_percent", snapshot.memory_percent, {})
            self.memory.log_metric("memory_used", snapshot.memory_used, {})
            self.memory.log_metric("active_threads", float(snapshot.active_threads), {})
            await asyncio.sleep(self.config.sample_interval_seconds)

    def start(self) -> None:
        if not self.config.enabled or self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Metrics collector started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def recent(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.memory.recent_metrics(limit=limit)
