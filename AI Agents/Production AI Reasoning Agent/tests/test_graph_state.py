from __future__ import annotations

from reasoning_agent.agent.state import AgentState


def test_agent_state_iteration_cap() -> None:
    state = AgentState(query="hello", max_iterations=2)

    assert state.should_continue() is True
    state.iteration = 2
    assert state.should_continue() is False
