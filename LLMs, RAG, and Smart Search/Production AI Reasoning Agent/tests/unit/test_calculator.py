from __future__ import annotations

from pathlib import Path

import pytest

from reasoning_agent.tooling.base import ToolContext
from reasoning_agent.tooling.tools.calculator import CalculatorInput, calculate


def test_calculator_arithmetic() -> None:
    out = calculate(
        CalculatorInput(expression="2 + 3 * 4"),
        ToolContext(session_id="s", run_id="r", workspace_root=Path(".")),
    )
    assert out.result == 14.0


def test_calculator_statistics() -> None:
    out = calculate(
        CalculatorInput(expression="mean([1, 2, 3, 4])"),
        ToolContext(session_id="s", run_id="r", workspace_root=Path(".")),
    )
    assert out.result == 2.5


def test_calculator_rejects_unsafe_expression() -> None:
    with pytest.raises(ValueError):
        calculate(
            CalculatorInput(expression="__import__('os').system('ls')"),
            ToolContext(session_id="s", run_id="r", workspace_root=Path(".")),
        )
