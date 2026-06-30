"""Registry for built-in and generated tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from api_intel_agent.tools.base import ToolResult

ToolHandler = Callable[..., Awaitable[ToolResult]]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    handler: ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def list_tools(self) -> list[dict[str, str]]:
        return [{"name": spec.name, "description": spec.description} for spec in self._tools.values()]

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    async def run(self, name: str, **kwargs: Any) -> ToolResult:
        spec = self._tools.get(name)
        if not spec:
            return ToolResult(name=name, success=False, payload={}, error="tool not found")
        return await spec.handler(**kwargs)
