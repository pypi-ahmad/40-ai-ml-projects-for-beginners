"""Tool registry factory."""

from __future__ import annotations

from api_intel_agent.tools.advanced_clients import graphql_query_tool, websocket_message_tool
from api_intel_agent.tools.builtins import (
    calculator_tool,
    chart_generator_tool,
    csv_export_tool,
    datetime_tool,
    file_reader_tool,
    http_client_tool,
    json_export_tool,
    markdown_generator_tool,
    pdf_report_tool,
    unit_conversion_tool,
    web_search_tool,
)
from api_intel_agent.tools.registry import ToolRegistry, ToolSpec


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolSpec(name="http_client", description="HTTP request tool", handler=http_client_tool))
    registry.register(ToolSpec(name="calculator", description="Safe calculator", handler=calculator_tool))
    registry.register(ToolSpec(name="file_reader", description="Read files in workspace", handler=file_reader_tool))
    registry.register(
        ToolSpec(name="markdown_generator", description="Generate markdown", handler=markdown_generator_tool)
    )
    registry.register(ToolSpec(name="csv_export", description="Export rows to CSV", handler=csv_export_tool))
    registry.register(ToolSpec(name="json_export", description="Export JSON payload", handler=json_export_tool))
    registry.register(ToolSpec(name="pdf_report", description="Generate PDF report", handler=pdf_report_tool))
    registry.register(ToolSpec(name="chart_generator", description="Generate plotly chart", handler=chart_generator_tool))
    registry.register(ToolSpec(name="web_search", description="Web search helper", handler=web_search_tool))
    registry.register(ToolSpec(name="datetime", description="Date and time", handler=datetime_tool))
    registry.register(
        ToolSpec(name="unit_conversion", description="Unit conversion helper", handler=unit_conversion_tool)
    )
    registry.register(ToolSpec(name="graphql_query", description="GraphQL query client", handler=graphql_query_tool))
    registry.register(
        ToolSpec(name="websocket_message", description="WebSocket client adapter", handler=websocket_message_tool)
    )
    return registry
