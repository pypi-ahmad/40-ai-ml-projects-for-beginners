"""Internal MCP-compatible server facade."""

from __future__ import annotations

from typing import Any

from crew_platform.mcp.contracts import MCPCallResponse, MCPToolSchema
from crew_platform.tools.registry import ToolRegistry


class InternalMCPServer:
    """Expose platform tools through MCP-like list/call API."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def list_tools(self) -> list[MCPToolSchema]:
        schemas: list[MCPToolSchema] = []
        for descriptor in self.registry.discover():
            schemas.append(
                MCPToolSchema(
                    name=descriptor.name,
                    description=descriptor.description,
                    input_schema=descriptor.input_schema,
                    output_schema=descriptor.output_schema,
                )
            )
        return schemas

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        run_id: str,
    ) -> MCPCallResponse:
        try:
            result = await self.registry.invoke(tool_name, payload=arguments, run_id=run_id)
            return MCPCallResponse(tool_name=tool_name, success=True, result=result)
        except Exception as exc:  # noqa: BLE001
            return MCPCallResponse(tool_name=tool_name, success=False, error=str(exc))
