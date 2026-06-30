from langgraph_platform.tools.builtin import CalculatorTool, UnitConverterTool


def test_calculator_basic() -> None:
    tool = CalculatorTool()
    result = tool.run({"expression": "2 + 3 * 4"})
    assert result.ok is True
    assert result.output == 14


def test_unit_converter() -> None:
    tool = UnitConverterTool()
    result = tool.run({"value": 1000, "from": "m", "to": "km"})
    assert result.ok is True
    assert result.output["value"] == 1
