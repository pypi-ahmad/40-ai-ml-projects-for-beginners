from __future__ import annotations

import logging
import time
from typing import Any

from tools.base import ToolDefinition

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> ToolDefinition:
        return self._tools[name]

    def list(self) -> list[dict[str, Any]]:
        return [tool.metadata() for tool in self._tools.values()]

    async def call(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            return {"ok": False, "error": f"Unknown tool: {name}"}

        tool = self._tools[name]
        started = time.perf_counter()

        try:
            result = await tool.handler(**payload)
            latency_ms = int((time.perf_counter() - started) * 1000)
            if "latency_ms" not in result:
                result["latency_ms"] = latency_ms
            return result
        except Exception as exc:
            logger.exception("Tool execution failed", extra={"tool": name})
            return {
                "ok": False,
                "error": str(exc),
                "tool": name,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
