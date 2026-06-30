"""State definitions for LangGraph workflow."""

from __future__ import annotations

from typing import Any, TypedDict

from reasoning_agent.schemas import IterationTrace, ReasoningMode, ToolObservation


class AgentState(TypedDict, total=False):
    """StateGraph state."""

    session_id: str
    run_id: str
    mode: ReasoningMode
    user_input: str
    objective: str
    plan_steps: list[str]
    current_step: int
    thought_summary: str
    selected_tool: str
    tool_args: dict[str, Any]
    last_observation_ok: bool
    last_error: str
    observation_summary: str
    observations: list[ToolObservation]
    trace: list[IterationTrace]
    answer: str
    citations: list[str]
    retries: int
    max_retries: int
    iteration: int
    max_iterations: int
    done: bool
    termination_reason: str
    errors: list[str]
    metrics: dict[str, float]
