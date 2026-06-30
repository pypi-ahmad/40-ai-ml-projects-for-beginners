"""Dynamic tool registry with invocation logging."""

from __future__ import annotations

import time
from typing import Any

from reasoning_agent.observability.events import EventRecord
from reasoning_agent.observability.metrics import MetricsStore
from reasoning_agent.observability.tracer import JsonlTracer
from reasoning_agent.tools.base import BaseTool, ToolDescriptor


class ToolRegistry:
    """Runtime registry for pluggable tools."""

    def __init__(self, tracer: JsonlTracer | None = None, metrics: MetricsStore | None = None) -> None:
        self._tools: dict[str, BaseTool[Any, Any]] = {}
        self._tracer = tracer
        self._metrics = metrics or MetricsStore()

    def register(self, tool: BaseTool[Any, Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool[Any, Any]:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def discover(self) -> list[ToolDescriptor]:
        return [tool.descriptor() for tool in self._tools.values()]

    async def invoke(self, name: str, payload: dict[str, Any], run_id: str) -> dict[str, Any]:
        tool = self.get(name)
        started = time.perf_counter()
        try:
            validated = tool.validate_input(payload)
            result = await tool.run(validated)
            validated_output = tool.validate_output(result)
            latency_ms = (time.perf_counter() - started) * 1000
            self._metrics.inc(f"tool.{name}.success")
            self._metrics.observe_ms(f"tool.{name}.latency_ms", latency_ms)
            self._log_event(
                run_id=run_id,
                status="ok",
                payload={
                    "tool": name,
                    "input": payload,
                    "output": validated_output.model_dump(mode="json"),
                    "latency_ms": latency_ms,
                },
            )
            out = validated_output.model_dump(mode="json")
            out.update({"tool": name, "latency_ms": latency_ms})
            return out
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - started) * 1000
            self._metrics.inc(f"tool.{name}.error")
            self._log_event(
                run_id=run_id,
                status="error",
                payload={
                    "tool": name,
                    "input": payload,
                    "error": str(exc),
                    "latency_ms": latency_ms,
                },
            )
            raise

    def _log_event(self, run_id: str, status: str, payload: dict[str, Any]) -> None:
        if self._tracer is None:
            return
        event = EventRecord(event_type="tool_call", run_id=run_id, status=status, payload=payload)
        self._tracer.safe_log(event)
