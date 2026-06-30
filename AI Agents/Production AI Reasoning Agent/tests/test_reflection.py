from __future__ import annotations

import pytest

from reasoning_agent.agent.reflection import Reflector
from reasoning_agent.agent.state import AgentState, PlanStep, ToolExecution


@pytest.mark.asyncio
async def test_reflector_retries_failed_step_with_budget() -> None:
    state = AgentState(
        query="calculate 2+2",
        plan=[PlanStep(step_id="1", description="math", tool_name="calculator", status="failed")],
        tool_calls=[
            ToolExecution(
                tool_name="calculator",
                input_payload={"expression": "2+2"},
                output_payload={},
                success=False,
                error="timeout",
            )
        ],
        error="timeout",
    )
    reflector = Reflector(max_retries=2, use_llm=False)

    updated = await reflector.reflect(state)

    assert updated.error is None
    assert updated.plan[0].status == "pending"
    assert updated.done is False


@pytest.mark.asyncio
async def test_reflector_stops_when_retry_budget_exhausted() -> None:
    state = AgentState(
        query="calculate 2+2",
        plan=[PlanStep(step_id="1", description="math", tool_name="calculator", status="failed")],
        tool_calls=[
            ToolExecution(
                tool_name="calculator",
                input_payload={"expression": "2+2"},
                output_payload={},
                success=False,
                error="boom",
            ),
            ToolExecution(
                tool_name="calculator",
                input_payload={"expression": "2+2"},
                output_payload={},
                success=False,
                error="boom",
            ),
        ],
        error="boom",
    )
    reflector = Reflector(max_retries=1, use_llm=False)

    updated = await reflector.reflect(state)

    assert updated.done is True
    assert updated.error == "boom"
