from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolExecutionRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    session_id: str = "api"


class WorkflowRequest(BaseModel):
    query: str


class ResourceReadRequest(BaseModel):
    uri: str


class PromptRenderRequest(BaseModel):
    prompt_name: str
    variables: dict[str, Any] = Field(default_factory=dict)
