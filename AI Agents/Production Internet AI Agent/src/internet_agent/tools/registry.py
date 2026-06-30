"""Runtime tool registry and invocation manager."""

from __future__ import annotations

import time
from typing import Any

from internet_agent.memory.repository import MemoryRepository
from internet_agent.metrics import METRICS
from internet_agent.tools.base import BaseTool, ToolDescriptor


class ToolRegistry:
    """Registry supporting dynamic registration and invocation telemetry."""

    def __init__(self, memory_repo: MemoryRepository) -> None:
        self._tools: dict[str, BaseTool[Any, Any]] = {}
        self._memory_repo = memory_repo

    def register(self, tool: BaseTool[Any, Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already exists: {tool.name}")
        self._tools[tool.name] = tool

    def discover(self) -> list[ToolDescriptor]:
        return [tool.descriptor() for tool in self._tools.values()]

    def has(self, name: str) -> bool:
        return name in self._tools

    async def invoke(self, session_id: str, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")

        tool = self._tools[name]
        started = time.perf_counter()
        status = "ok"
        result: dict[str, Any]

        try:
            validated = tool.validate_input(payload)
            output = await tool.run(validated)
            validated_output = tool.validate_output(output.model_dump(mode="json"))
            result = validated_output.model_dump(mode="json")
            METRICS.inc(f"tool.{name}.success")
        except Exception as exc:  # noqa: BLE001
            status = "error"
            result = {"error": str(exc)}
            METRICS.inc(f"tool.{name}.error")
            raise
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            METRICS.observe_ms(f"tool.{name}.latency_ms", latency_ms)
            self._memory_repo.add_tool_history(
                session_id=session_id,
                tool_name=name,
                tool_input=payload,
                tool_output=result,
                status=status,
                latency_ms=latency_ms,
            )

        return {**result, "tool": name}
