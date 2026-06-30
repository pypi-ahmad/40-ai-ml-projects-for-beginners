"""MCP-style tool adapter abstraction for external tool servers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MCPToolCall:
    server: str
    tool: str
    arguments: dict[str, Any]


class MCPClientAdapter:
    """Minimal integration point for MCP tool servers.

    This project keeps transport pluggable so local/remote MCP clients can be
    injected without changing graph logic.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    async def call(self, request: MCPToolCall) -> dict[str, Any]:
        if not self.enabled:
            return {
                "status": "disabled",
                "server": request.server,
                "tool": request.tool,
                "result": None,
            }
        # Hook point for real MCP transport.
        return {
            "status": "not_implemented",
            "server": request.server,
            "tool": request.tool,
            "arguments": request.arguments,
        }
