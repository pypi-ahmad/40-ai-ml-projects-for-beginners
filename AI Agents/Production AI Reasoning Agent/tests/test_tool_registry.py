from __future__ import annotations

import pytest
from pydantic import BaseModel

from reasoning_agent.tools.base import BaseTool
from reasoning_agent.tools.registry import ToolRegistry


class DummyInput(BaseModel):
    value: int


class DummyOutput(BaseModel):
    result: int


class DummyTool(BaseTool[DummyInput, DummyOutput]):
    name = "dummy"
    description = "dummy tool"
    input_model = DummyInput
    output_model = DummyOutput

    async def run(self, payload: DummyInput) -> DummyOutput:
        return DummyOutput(result=payload.value + 1)


def test_register_and_discover_tool() -> None:
    registry = ToolRegistry()
    registry.register(DummyTool())

    discovered = registry.discover()

    assert len(discovered) == 1
    assert discovered[0].name == "dummy"


@pytest.mark.asyncio
async def test_validate_and_invoke_tool() -> None:
    registry = ToolRegistry()
    registry.register(DummyTool())

    output = await registry.invoke("dummy", {"value": 4}, run_id="run-1")

    assert output["result"] == 5
    assert output["tool"] == "dummy"


def test_duplicate_tool_registration_fails() -> None:
    registry = ToolRegistry()
    registry.register(DummyTool())

    with pytest.raises(ValueError):
        registry.register(DummyTool())
