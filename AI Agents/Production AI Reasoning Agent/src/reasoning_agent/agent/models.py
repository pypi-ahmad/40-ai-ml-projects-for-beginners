"""Request/response models for public runner interface."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """Incoming request for agent execution."""

    query: str
    session_id: str = "default"
    reasoning_mode: str | None = None


class AgentRunResult(BaseModel):
    """Final result and trace bundle."""

    answer: str
    success: bool
    plan: list[dict[str, object]] = Field(default_factory=list)
    tool_calls: list[dict[str, object]] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    reflection: str = ""
    error: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)
