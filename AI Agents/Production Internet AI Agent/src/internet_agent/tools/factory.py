"""Factory for default registry composition."""

from __future__ import annotations

from internet_agent.memory.repository import MemoryRepository
from internet_agent.tools.local_tools import CSVReaderTool, FileReaderTool, JSONReaderTool
from internet_agent.tools.registry import ToolRegistry
from internet_agent.tools.search_tools import (
    DuckDuckGoSearchTool,
    GitHubSearchTool,
    HtmlCleanerTool,
    MarkdownExtractorTool,
    NewsSearchTool,
    PDFReaderTool,
    WebsiteFetcherTool,
    WikipediaSearchTool,
)
from internet_agent.tools.utility_tools import (
    CalculatorTool,
    CurrencyExchangeTool,
    DateTimeTool,
    PythonCalculatorTool,
    UnitConverterTool,
    WeatherTool,
)


def build_default_registry(memory_repo: MemoryRepository) -> ToolRegistry:
    registry = ToolRegistry(memory_repo)

    for tool in (
        DuckDuckGoSearchTool(),
        WebsiteFetcherTool(),
        HtmlCleanerTool(),
        MarkdownExtractorTool(),
        PDFReaderTool(),
        CalculatorTool(),
        UnitConverterTool(),
        WeatherTool(),
        CurrencyExchangeTool(),
        GitHubSearchTool(),
        NewsSearchTool(),
        WikipediaSearchTool(),
        FileReaderTool(),
        CSVReaderTool(),
        JSONReaderTool(),
        PythonCalculatorTool(),
        DateTimeTool(),
    ):
        registry.register(tool)

    return registry
