"""MCP compatibility contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MCPToolSchema(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


class MCPCallRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class MCPCallResponse(BaseModel):
    tool_name: str
    success: bool
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
