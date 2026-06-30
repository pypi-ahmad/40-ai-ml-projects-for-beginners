"""MCP exports."""

from crew_platform.mcp.client import ExternalMCPClient
from crew_platform.mcp.contracts import MCPCallRequest, MCPCallResponse, MCPToolSchema
from crew_platform.mcp.server import InternalMCPServer

__all__ = ["ExternalMCPClient", "InternalMCPServer", "MCPCallRequest", "MCPCallResponse", "MCPToolSchema"]
