from __future__ import annotations

import pytest

from reasoning_agent.tools.python_repl import PythonInput, PythonREPLTool


@pytest.mark.asyncio
async def test_python_tool_executes_safe_code() -> None:
    tool = PythonREPLTool(timeout_seconds=2, memory_limit_mb=64)

    result = await tool.run(PythonInput(code="print(sum([1,2,3]))"))

    assert result.stdout.strip() == "6"
    assert result.success is True


@pytest.mark.asyncio
async def test_python_tool_blocks_forbidden_imports() -> None:
    tool = PythonREPLTool(timeout_seconds=2, memory_limit_mb=64)

    with pytest.raises(ValueError):
        await tool.run(PythonInput(code="import os\nprint(os.listdir('.'))"))


@pytest.mark.asyncio
async def test_python_tool_blocks_dunder_attribute_access() -> None:
    tool = PythonREPLTool(timeout_seconds=2, memory_limit_mb=64)

    with pytest.raises(ValueError):
        await tool.run(PythonInput(code="print((1).__class__.__mro__)"))


@pytest.mark.asyncio
async def test_python_tool_blocks_getattr_escape() -> None:
    tool = PythonREPLTool(timeout_seconds=2, memory_limit_mb=64)

    with pytest.raises(ValueError):
        await tool.run(PythonInput(code="print(getattr(1, '__class__'))"))


@pytest.mark.asyncio
async def test_python_tool_timeout_returns_failure() -> None:
    tool = PythonREPLTool(timeout_seconds=1, memory_limit_mb=64)

    result = await tool.run(PythonInput(code="while True:\n    pass"))

    assert result.success is False
    assert result.return_code == 124
