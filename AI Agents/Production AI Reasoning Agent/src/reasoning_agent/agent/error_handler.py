"""Error normalization and graceful handling."""

from __future__ import annotations

from dataclasses import dataclass

from reasoning_agent.agent.state import AgentState


@dataclass(slots=True)
class ErrorHandler:
    def apply(self, state: AgentState) -> AgentState:
        if state.error is None:
            return state
        state.thoughts.append(f"Error handled: {state.error}")
        if state.iteration >= state.max_iterations:
            state.done = True
        return state
