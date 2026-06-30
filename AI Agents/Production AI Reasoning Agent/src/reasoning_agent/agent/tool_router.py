"""Tool selection and routing."""

from __future__ import annotations

from dataclasses import dataclass

from reasoning_agent.agent.state import AgentState
from reasoning_agent.tools.registry import ToolRegistry


@dataclass(slots=True)
class ToolRouter:
    registry: ToolRegistry

    def route(self, state: AgentState) -> AgentState:
        for step in state.plan:
            if step.status != "pending":
                continue
            if step.tool_name is None:
                step.status = "ready_no_tool"
                return state
            try:
                self.registry.get(step.tool_name)
                step.status = "ready"
                return state
            except KeyError:
                step.status = "failed"
                state.error = f"Requested tool not found: {step.tool_name}"
                return state
        state.done = True
        return state
