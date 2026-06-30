"""Typed state objects for LangGraph orchestration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    step_id: str
    description: str
    tool_name: str | None = None
    tool_input: dict[str, object] = Field(default_factory=dict)
    status: str = "pending"


class ToolExecution(BaseModel):
    tool_name: str
    input_payload: dict[str, object]
    output_payload: dict[str, object] = Field(default_factory=dict)
    success: bool = True
    error: str | None = None


class AgentState(BaseModel):
    """State propagated across graph nodes."""

    query: str
    reasoning_mode: str = "react"
    max_iterations: int = 10
    iteration: int = 0
    retry_count: int = 0
    thoughts: list[str] = Field(default_factory=list)
    plan: list[PlanStep] = Field(default_factory=list)
    tool_calls: list[ToolExecution] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    reflection: str = ""
    final_answer: str = ""
    done: bool = False
    error: str | None = None
    citations: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)

    def should_continue(self) -> bool:
        return not self.done and self.iteration < self.max_iterations and self.error is None
