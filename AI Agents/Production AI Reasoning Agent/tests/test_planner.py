from __future__ import annotations

import pytest

from reasoning_agent.agent.planner import Planner
from reasoning_agent.agent.state import AgentState


class BrokenLLM:
    async def generate(self, **_: object):
        raise RuntimeError("offline")


class ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    async def generate(self, **_: object):
        class R:
            def __init__(self, text: str) -> None:
                self.text = text

        if not self.responses:
            raise RuntimeError("no response")
        return R(self.responses.pop(0))


@pytest.mark.asyncio
async def test_planner_fallback_math() -> None:
    planner = Planner(llm=BrokenLLM(), model="qwen3:8b", temperature=0.1, max_tokens=128)
    state = AgentState(query="calculate 2+2")

    out = await planner.build_plan(state, available_tools=["calculator"])

    assert out.plan
    assert out.plan[0].tool_name == "calculator"
    assert out.plan[0].tool_input["expression"] == "2+2"


@pytest.mark.asyncio
async def test_planner_fallback_unit_conversion() -> None:
    planner = Planner(llm=BrokenLLM(), model="qwen3:8b", temperature=0.1, max_tokens=128)
    state = AgentState(query="Convert 12 kilometers to miles")

    out = await planner.build_plan(state, available_tools=["unit_converter"])

    assert out.plan[0].tool_name == "unit_converter"
    assert out.plan[0].tool_input["value"] == 12.0


@pytest.mark.asyncio
async def test_planner_repairs_unavailable_tool_with_selector() -> None:
    llm = ScriptedLLM(
        responses=[
            (
                '{"thoughts":["plan"],"steps":[{"step_id":"1","description":"Solve expression",'
                '"tool_name":"bad_tool","tool_input":{"expression":"2+2"}}]}'
            ),
            '{"tool_name":"calculator","reason":"math expression detected"}',
        ]
    )
    planner = Planner(llm=llm, model="qwen3:8b", temperature=0.1, max_tokens=128)
    state = AgentState(query="calculate 2+2")

    out = await planner.build_plan(state, available_tools=["calculator", "duckduckgo_search"])

    assert out.plan[0].tool_name == "calculator"
