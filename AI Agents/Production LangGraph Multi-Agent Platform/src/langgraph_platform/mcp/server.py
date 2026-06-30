"""Internal MCP-exposed tools service."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from langgraph_platform.tools.base import ToolRegistry


class MCPToolRequest(BaseModel):
    """Incoming MCP-like tool execution request."""

    tool: str
    args: dict[str, Any] = {}


class MCPServerAdapter:
    """Expose selected tools for external MCP clients."""

    def __init__(self, registry: ToolRegistry, exposed_tools: list[str]) -> None:
        self.registry = registry
        self.exposed_tools = set(exposed_tools)

    def execute(self, request: MCPToolRequest) -> dict[str, Any]:
        if request.tool not in self.exposed_tools:
            return {"ok": False, "error": f"Tool not exposed: {request.tool}"}
        result = self.registry.run(request.tool, request.args)
        return {
            "ok": result.ok,
            "output": result.output,
            "source": result.source,
            "error": result.error,
        }

    def router(self) -> APIRouter:
        """FastAPI router exposing MCP-like endpoints."""

        router = APIRouter(prefix="/mcp", tags=["mcp"])

        @router.get("/capabilities")
        def capabilities() -> dict[str, Any]:
            return {"tools": sorted(self.exposed_tools)}

        @router.post("/call")
        def call(request: MCPToolRequest) -> dict[str, Any]:
            return self.execute(request)

        return router
