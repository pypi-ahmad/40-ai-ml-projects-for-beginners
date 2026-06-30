from __future__ import annotations

import pytest

from config.settings import load_settings
from memory.service import MemoryService
from prompts.library import PromptLibrary
from resources.library import ResourceLibrary
from tools.builtin import build_builtin_tools
from tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_builtin_tool_registry_contains_required_tools() -> None:
    settings = load_settings("configs/default.yaml")
    memory = MemoryService(settings)
    registry = ToolRegistry()
    resources = ResourceLibrary(settings, memory)
    prompts = PromptLibrary()

    for tool in build_builtin_tools(settings=settings, memory=memory, resources=resources, prompts=prompts):
        registry.register(tool)

    expected = {
        "calculator",
        "weather",
        "file_reader",
        "file_writer",
        "csv_reader",
        "json_reader",
        "sqlite_query",
        "chroma_search",
        "web_search",
        "github_search",
        "news_search",
        "python_executor",
        "markdown_generator",
        "report_generator",
        "pdf_generator",
        "directory_search",
        "code_search",
        "shell_command",
        "system_information",
    }

    assert expected.issubset(set(registry.names()))

    calc_result = await registry.call("calculator", {"expression": "2 + 3 * 4"})
    assert calc_result["ok"] is True
    assert calc_result["result"] == 14
