"""Built-in tool registration."""

from __future__ import annotations

from reasoning_agent.memory import MemoryService
from reasoning_agent.settings import Settings
from reasoning_agent.tooling.registry import ToolRegistry
from reasoning_agent.tooling.tools import (
    calculator,
    csv_analyzer,
    currency_converter,
    datetime_tool,
    document_search,
    duckduckgo_search,
    file_reader,
    json_explorer,
    local_rag,
    markdown_reader,
    optional_tools,
    python_repl,
    semantic_search,
    unit_converter,
    vector_search,
    weather_tool,
    webpage_reader,
    wikipedia_tool,
)


def register_default_tools(registry: ToolRegistry, settings: Settings, memory: MemoryService) -> None:
    """Register required and optional tools into registry."""

    registry.register(calculator.spec, calculator.calculate)
    registry.register(duckduckgo_search.spec, duckduckgo_search.search)
    registry.register(wikipedia_tool.spec, wikipedia_tool.wikipedia_lookup)
    registry.register(python_repl.spec, python_repl.run_python)
    registry.register(datetime_tool.spec, datetime_tool.current_datetime)
    registry.register(unit_converter.spec, unit_converter.convert_units)

    weather_provider = weather_tool.WeatherProvider(
        provider=settings.weather_provider,
        api_key=settings.weather_api_key,
    )
    currency_provider = currency_converter.CurrencyProvider(
        provider=settings.currency_provider,
        api_key=settings.currency_api_key,
    )

    registry.register(weather_tool.spec, weather_tool.make_handler(weather_provider))
    registry.register(currency_converter.spec, currency_converter.make_handler(currency_provider))
    registry.register(file_reader.spec, file_reader.read_file)
    registry.register(csv_analyzer.spec, csv_analyzer.analyze_csv)
    registry.register(json_explorer.spec, json_explorer.explore_json)
    registry.register(webpage_reader.spec, webpage_reader.read_webpage)
    registry.register(markdown_reader.spec, markdown_reader.read_markdown)
    registry.register(document_search.spec, document_search.search_documents)
    registry.register(semantic_search.spec, semantic_search.make_handler(memory))
    registry.register(vector_search.spec, vector_search.make_handler(memory))
    registry.register(local_rag.spec, local_rag.make_handler(memory))

    registry.register(optional_tools.sqlite_spec, optional_tools.unavailable)
    registry.register(optional_tools.github_spec, optional_tools.unavailable)
    registry.register(optional_tools.news_spec, optional_tools.unavailable)
    registry.register(optional_tools.arxiv_spec, optional_tools.unavailable)
