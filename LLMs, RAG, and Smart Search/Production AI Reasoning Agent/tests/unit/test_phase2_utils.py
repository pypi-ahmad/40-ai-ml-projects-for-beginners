from __future__ import annotations

import asyncio
from pathlib import Path

from reasoning_agent.adapters.mcp_adapter import MCPToolAdapter
from reasoning_agent.agent.approval import ApprovalPolicy
from reasoning_agent.response.streaming import stream_text
from reasoning_agent.tooling import ToolContext, ToolRegistry, ToolSpec
from reasoning_agent.tooling.async_runner import AsyncToolRunner


class _In:
    @staticmethod
    def model_validate(payload):
        class P:
            def __init__(self, x):
                self.x = x

        return P(payload["x"])

    @staticmethod
    def model_json_schema():
        return {"type": "object"}


class _Out:
    def __init__(self, y):
        self.y = y

    def model_dump(self):
        return {"y": self.y}

    @staticmethod
    def model_validate(payload):
        return _Out(payload["y"])

    @staticmethod
    def model_json_schema():
        return {"type": "object"}


def _handler(payload, _):
    return _Out(payload.x + 1)


def test_mcp_adapter_and_approval_and_streaming(tmp_path: Path) -> None:
    registry = ToolRegistry(workspace_root=tmp_path)
    registry.register(ToolSpec("inc", "increment", _In, _Out), _handler)

    adapter = MCPToolAdapter(registry)
    tools = adapter.list_tools()
    assert tools[0]["name"] == "inc"

    policy = ApprovalPolicy()
    assert policy.evaluate("python_repl", {"code": "print(1)"}).required is True
    assert policy.evaluate("calculator", {}).required is False

    chunks = list(stream_text("abcdef", chunk_size=2))
    assert chunks == ["ab", "cd", "ef"]


def test_async_tool_runner(tmp_path: Path) -> None:
    registry = ToolRegistry(workspace_root=tmp_path)
    registry.register(ToolSpec("inc", "increment", _In, _Out), _handler)
    runner = AsyncToolRunner(registry)

    async def _run():
        ctx = ToolContext("s", "r", tmp_path)
        rows = await runner.invoke_many([("inc", {"x": 1}), ("inc", {"x": 2})], ctx)
        return rows

    out = asyncio.run(_run())
    assert out[0]["ok"] is True
    assert out[1]["output"] == {"y": 3}
