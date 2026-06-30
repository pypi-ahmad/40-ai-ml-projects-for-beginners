"""External MCP client adapter."""

from __future__ import annotations

from typing import Any

import httpx


class ExternalMCPClient:
    """Lightweight adapter for external MCP HTTP bridges."""

    def __init__(self, endpoint: str, timeout_seconds: int = 20) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def list_tools(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(f"{self.endpoint}/tools")
            response.raise_for_status()
            data = response.json()
        return data.get("tools", [])

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = {"tool_name": tool_name, "arguments": arguments}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.endpoint}/call", json=payload)
            response.raise_for_status()
            return response.json()
