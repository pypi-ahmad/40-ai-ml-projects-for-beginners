"""Tool abstractions and plugin-friendly registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class ToolContext:
    """Shared context for tool execution."""

    workflow_id: str
    session_id: str


@dataclass(slots=True)
class ToolResult:
    """Standardized tool response."""

    ok: bool
    output: Any
    source: str = ""
    error: str | None = None


class Tool(Protocol):
    """Protocol for pluggable tools."""

    name: str

    def run(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        """Run tool with typed args and context."""


class ToolRegistry:
    """Registry for built-in and plugin tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]

    def run(
        self, name: str, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        return self.get(name).run(args=args, context=context)

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def close(self) -> None:
        """Close tool resources when tools expose a close hook."""

        for tool in self._tools.values():
            close_fn = getattr(tool, "close", None)
            if callable(close_fn):
                close_fn()
