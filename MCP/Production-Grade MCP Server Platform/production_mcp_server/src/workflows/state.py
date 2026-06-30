from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class WorkflowState(BaseModel):
    query: str
    plan: list[str] = Field(default_factory=list)
    selected_tools: list[str] = Field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = Field(default_factory=list)
    memory_context: list[dict[str, Any]] = Field(default_factory=list)
    reflection: str = ""
    report: str = ""
    steps: list[str] = Field(default_factory=list)
    status: Literal["running", "completed", "degraded", "failed"] = "running"
