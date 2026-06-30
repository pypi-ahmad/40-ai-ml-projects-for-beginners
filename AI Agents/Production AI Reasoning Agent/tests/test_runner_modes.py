from __future__ import annotations

import pytest

from reasoning_agent.agent.models import AgentRequest
from reasoning_agent.agent.runner import AgentRunner
from reasoning_agent.config import Settings


class BrokenGraph:
    async def ainvoke(self, _: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("graph exploded")


class SlowGraph:
    async def ainvoke(self, _: dict[str, object]) -> dict[str, object]:
        import asyncio

        await asyncio.sleep(1)
        return {"state": {}}


def _offline_settings() -> Settings:
    settings = Settings()
    settings.memory.chroma_enabled = False
    settings.agent.use_llm_for_planning = False
    settings.agent.use_llm_for_response = False
    settings.tools.enable_python_tool = False
    settings.tools.enabled_tools = ["calculator", "datetime", "unit_converter", "currency_converter"]
    settings.agent.max_iterations = 4
    return settings


@pytest.mark.asyncio
async def test_runner_fallback_mode_executes_plan() -> None:
    settings = _offline_settings()
    settings.agent.runtime_mode = "fallback"

    runner = AgentRunner(settings=settings)
    result = await runner.run(AgentRequest(query="calculate 2+2", session_id="test-fallback"))

    assert result.success is True
    assert "4" in result.answer
    assert result.tool_calls


@pytest.mark.asyncio
async def test_runner_graph_mode_can_fallback_on_error() -> None:
    settings = _offline_settings()
    settings.agent.runtime_mode = "graph"
    settings.agent.graph_fallback_on_error = True

    runner = AgentRunner(settings=settings)
    runner.graph = BrokenGraph()
    result = await runner.run(AgentRequest(query="calculate 3+5", session_id="test-graph-fallback"))

    assert result.success is True
    assert "8" in result.answer
    assert result.metrics.get("graph.error", 0.0) >= 1.0


@pytest.mark.asyncio
async def test_runner_graph_mode_without_fallback_returns_failure() -> None:
    settings = _offline_settings()
    settings.agent.runtime_mode = "graph"
    settings.agent.graph_fallback_on_error = False

    runner = AgentRunner(settings=settings)
    runner.graph = BrokenGraph()
    result = await runner.run(AgentRequest(query="calculate 3+5", session_id="test-graph-strict"))

    assert result.success is False
    assert result.error is not None
    assert "LangGraph execution failed" in result.error


@pytest.mark.asyncio
async def test_runner_graph_timeout_falls_back() -> None:
    settings = _offline_settings()
    settings.agent.runtime_mode = "graph"
    settings.agent.graph_fallback_on_error = True
    settings.agent.graph_timeout_seconds = 0

    runner = AgentRunner(settings=settings)
    runner.graph = SlowGraph()
    result = await runner.run(AgentRequest(query="calculate 9+1", session_id="test-graph-timeout"))

    assert result.success is True
    assert "10" in result.answer
    assert result.metrics.get("graph.timeout", 0.0) >= 1.0
