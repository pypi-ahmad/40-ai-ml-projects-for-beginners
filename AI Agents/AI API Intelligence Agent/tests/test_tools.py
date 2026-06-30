import pytest

from api_intel_agent.tools.factory import build_tool_registry


@pytest.mark.asyncio
async def test_calculator_tool():
    tools = build_tool_registry()
    result = await tools.run('calculator', expression='2 + 3 * 4')
    assert result.success
    assert result.payload['result'] == 14
