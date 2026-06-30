"""MCP client adapter for external servers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MCPServerConfig:
    """External MCP server configuration."""

    name: str
    transport: str = "stdio"
    endpoint: str | None = None


class MCPClient:
    """Minimal MCP client adapter abstraction."""

    def __init__(self, servers: list[MCPServerConfig] | None = None) -> None:
        self.servers = servers or []

    def discover_capabilities(self) -> dict[str, list[str]]:
        """Return capability placeholders for configured servers."""

        return {
            server.name: ["search", "calculator", "documents", "memory"] for server in self.servers
        }

    def call(self, server_name: str, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch tool call to external MCP server.

        This implementation is intentionally transport-agnostic and can be swapped
        with a full MCP SDK adapter.
        """

        if server_name not in {server.name for server in self.servers}:
            raise KeyError(f"Unknown MCP server: {server_name}")
        return {
            "server": server_name,
            "tool": tool_name,
            "payload": payload,
            "status": "not_implemented_transport_stub",
        }
