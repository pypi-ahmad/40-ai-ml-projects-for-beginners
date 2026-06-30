from __future__ import annotations

import pytest

from reasoning_agent.tools.calculator import CalculatorInput, CalculatorTool


@pytest.mark.asyncio
async def test_calculator_arithmetic_expression() -> None:
    tool = CalculatorTool()

    result = await tool.run(CalculatorInput(expression="(2 + 3) * 5"))

    assert result.value == 25


@pytest.mark.asyncio
async def test_calculator_statistics_mean() -> None:
    tool = CalculatorTool()

    result = await tool.run(CalculatorInput(expression="mean(1,2,3,4,5)"))

    assert result.value == 3


@pytest.mark.asyncio
async def test_calculator_blocks_unsafe_names() -> None:
    tool = CalculatorTool()

    with pytest.raises(ValueError):
        await tool.run(CalculatorInput(expression="__import__('os').system('id')"))
