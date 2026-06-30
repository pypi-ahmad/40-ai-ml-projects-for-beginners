from __future__ import annotations

import pytest

from internet_agent.tools.utility_tools import (
    CalculatorInput,
    CalculatorTool,
    UnitConverterTool,
    UnitInput,
)


@pytest.mark.asyncio
async def test_calculator_tool() -> None:
    tool = CalculatorTool()
    result = await tool.run(CalculatorInput(expression="2 + 3 * 4"))
    assert result.result == 14.0


@pytest.mark.asyncio
async def test_unit_converter_tool() -> None:
    tool = UnitConverterTool()
    result = await tool.run(UnitInput(value=1, from_unit="km", to_unit="m"))
    assert result.result == 1000.0
