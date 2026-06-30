"""External MCP hook registry for integrating third-party tool calls."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ExternalTool = Callable[[dict[str, Any]], dict[str, Any]]


class ExternalMCPHookRegistry:
    """Register and invoke external MCP-compatible tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ExternalTool] = {}

    def register(self, tool_name: str, handler: ExternalTool) -> None:
        """Register external tool handler."""

        self._tools[tool_name] = handler

    def call(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Invoke registered external tool."""

        if tool_name not in self._tools:
            return {
                "ok": False,
                "error": f"external_tool_not_registered:{tool_name}",
            }
        return self._tools[tool_name](payload)

    def available(self) -> list[str]:
        """List registered external tools."""

        return sorted(self._tools.keys())
