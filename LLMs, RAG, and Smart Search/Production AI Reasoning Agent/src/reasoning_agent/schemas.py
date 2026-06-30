"""Shared schemas for reasoning agent."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ReasoningMode(str, Enum):
    """Supported reasoning execution modes."""

    REACT = "react"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    SELF_REFLECTION = "self_reflection"
    RETRY = "retry"
    TREE_OF_THOUGHT = "tree_of_thought"


class ToolCall(BaseModel):
    """Single tool invocation request."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolObservation(BaseModel):
    """Single tool invocation result."""

    tool: str
    ok: bool
    output: Any | None = None
    error: str | None = None
    latency_ms: float = 0.0
    citations: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class IterationTrace(BaseModel):
    """One iteration trace summary."""

    iteration: int
    thought_summary: str
    action: ToolCall | None = None
    observation: ToolObservation | None = None
    reflection: str = ""
    retries: int = 0
    latency_ms: float = 0.0


class AgentResponse(BaseModel):
    """Final response contract."""

    session_id: str
    run_id: str
    answer: str
    mode: ReasoningMode
    success: bool
    iterations: int
    termination_reason: Literal[
        "completed",
        "max_iterations",
        "tool_failure",
        "parser_failure",
        "timeout",
        "error",
    ]
    trace: list[IterationTrace] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PlanningOutput(BaseModel):
    """Planner structured output."""

    objective: str
    steps: list[str]
    reasoning_summary: str
    required_tools: list[str] = Field(default_factory=list)


class ToolRoutingOutput(BaseModel):
    """Tool routing structured output."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    justification: str


class ReflectionOutput(BaseModel):
    """Reflection structured output."""

    success: bool
    confidence: float = Field(ge=0.0, le=1.0)
    revised_plan: list[str] = Field(default_factory=list)
    notes: str = ""


class FinalAnswerOutput(BaseModel):
    """Response generation structured output."""

    answer: str
    citations: list[str] = Field(default_factory=list)
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
