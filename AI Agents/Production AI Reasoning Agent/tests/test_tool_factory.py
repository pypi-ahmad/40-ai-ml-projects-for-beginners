from __future__ import annotations

from pathlib import Path

from reasoning_agent.observability.metrics import MetricsStore
from reasoning_agent.tools.factory import create_default_registry


def test_factory_registers_required_tools() -> None:
    registry = create_default_registry(
        workspace_root=Path("."),
        tracer=None,
        metrics=MetricsStore(),
        memory_store=None,
        enable_python_tool=True,
    )

    names = {tool.name for tool in registry.discover()}
    assert "calculator" in names
    assert "duckduckgo_search" in names
    assert "python_repl" in names
    assert "local_rag" in names


def test_factory_disables_python_tool_by_default() -> None:
    registry = create_default_registry(
        workspace_root=Path("."),
        tracer=None,
        metrics=MetricsStore(),
        memory_store=None,
    )

    names = {tool.name for tool in registry.discover()}
    assert "python_repl" not in names


def test_factory_respects_enabled_tools_filter() -> None:
    registry = create_default_registry(
        workspace_root=Path("."),
        tracer=None,
        metrics=MetricsStore(),
        memory_store=None,
        enabled_tools=["calculator", "datetime"],
        enable_python_tool=True,
    )

    names = {tool.name for tool in registry.discover()}
    assert names == {"calculator", "datetime"}
