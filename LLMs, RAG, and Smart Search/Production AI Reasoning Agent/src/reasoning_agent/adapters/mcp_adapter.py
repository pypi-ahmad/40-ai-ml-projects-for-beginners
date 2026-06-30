"""MCP-compatible adapter for tool registry metadata."""

from __future__ import annotations

from typing import Any

from reasoning_agent.tooling import ToolRegistry


class MCPToolAdapter:
    """Expose local tools in MCP-like schema contract."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def list_tools(self) -> list[dict[str, Any]]:
        """Return MCP-style tool descriptors."""

        tools: list[dict[str, Any]] = []
        for spec in self.registry.specs():
            tools.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "inputSchema": spec.input_schema,
                    "outputSchema": spec.output_schema,
                    "metadata": {
                        "tags": spec.tags,
                        "timeout_s": spec.timeout_s,
                        "requires_network": spec.requires_network,
                    },
                }
            )
        return tools
