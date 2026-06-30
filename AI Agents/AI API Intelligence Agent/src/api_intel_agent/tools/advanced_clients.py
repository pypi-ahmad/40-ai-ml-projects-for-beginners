"""GraphQL and WebSocket helper clients for advanced API support."""

from __future__ import annotations

from typing import Any

import httpx

from api_intel_agent.tools.base import ToolResult


async def graphql_query_tool(
    endpoint: str,
    query: str,
    variables: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> ToolResult:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                endpoint,
                json={"query": query, "variables": variables or {}},
                headers=headers,
            )
            response.raise_for_status()
            return ToolResult(name="graphql_query", success=True, payload=response.json())
    except Exception as exc:
        return ToolResult(name="graphql_query", success=False, payload={}, error=str(exc))


async def websocket_message_tool(url: str, message: str) -> ToolResult:
    # Lightweight HTTP fallback for environments without websocket extras.
    # Caller can route to specialized websocket implementation when installed.
    _ = url
    _ = message
    return ToolResult(
        name="websocket_message",
        success=False,
        payload={},
        error="websocket client extras not installed in default build",
    )
