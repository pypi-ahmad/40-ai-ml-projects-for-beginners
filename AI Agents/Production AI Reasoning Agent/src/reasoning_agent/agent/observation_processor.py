"""Observation processor for post-tool reasoning."""

from __future__ import annotations

from dataclasses import dataclass

from reasoning_agent.agent.state import AgentState


@dataclass(slots=True)
class ObservationProcessor:
    def process(self, state: AgentState) -> AgentState:
        if state.observations:
            latest = state.observations[-1]
            state.thoughts.append(f"Observed: {latest[:200]}")
        state.iteration += 1
        pending = [step for step in state.plan if step.status == "pending"]
        failed = [step for step in state.plan if step.status == "failed"]
        if not pending and not failed and state.error is None:
            state.done = True
        return state
