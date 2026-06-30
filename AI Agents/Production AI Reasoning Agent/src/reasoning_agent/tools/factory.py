"""Factory methods for dynamic tool registration."""

from __future__ import annotations

from pathlib import Path

from reasoning_agent.memory.chroma_store import ChromaSemanticStore
from reasoning_agent.observability.metrics import MetricsStore
from reasoning_agent.observability.tracer import JsonlTracer
from reasoning_agent.tools.calculator import CalculatorTool
from reasoning_agent.tools.arxiv_search import ArxivSearchTool
from reasoning_agent.tools.currency_converter import CurrencyConverterTool
from reasoning_agent.tools.csv_analyzer import CSVAnalyzerTool
from reasoning_agent.tools.datetime_tool import DatetimeTool
from reasoning_agent.tools.document_search import DocumentSearchTool
from reasoning_agent.tools.file_reader import FileReaderTool
from reasoning_agent.tools.github_search import GitHubSearchTool
from reasoning_agent.tools.json_explorer import JSONExplorerTool
from reasoning_agent.tools.local_rag import LocalRAGTool
from reasoning_agent.tools.markdown_reader import MarkdownReaderTool
from reasoning_agent.tools.news_search import NewsSearchTool
from reasoning_agent.tools.python_repl import PythonREPLTool
from reasoning_agent.tools.registry import ToolRegistry
from reasoning_agent.tools.search import DuckDuckGoSearchTool
from reasoning_agent.tools.sqlite_tool import SQLiteQueryTool
from reasoning_agent.tools.semantic_search import SemanticSearchTool
from reasoning_agent.tools.unit_converter import UnitConverterTool
from reasoning_agent.tools.vector_search import VectorSearchTool
from reasoning_agent.tools.weather import WeatherTool
from reasoning_agent.tools.webpage_reader import WebpageReaderTool
from reasoning_agent.tools.wikipedia_tool import WikipediaTool


def create_default_registry(
    workspace_root: Path,
    tracer: JsonlTracer | None,
    metrics: MetricsStore,
    memory_store: ChromaSemanticStore | None = None,
    python_timeout: int = 5,
    python_memory_mb: int = 128,
    optional_tools: list[str] | None = None,
    enabled_tools: list[str] | None = None,
    enable_python_tool: bool = False,
) -> ToolRegistry:
    """Create and pre-register all required tools."""

    registry = ToolRegistry(tracer=tracer, metrics=metrics)
    selected_required = set(enabled_tools or ["*"])

    def is_enabled(name: str) -> bool:
        return "*" in selected_required or name in selected_required

    def register_if_enabled(tool_name: str, factory) -> None:
        if not is_enabled(tool_name):
            return
        registry.register(factory())

    register_if_enabled("calculator", CalculatorTool)
    register_if_enabled("duckduckgo_search", DuckDuckGoSearchTool)
    register_if_enabled("wikipedia", WikipediaTool)
    if enable_python_tool and is_enabled("python_repl"):
        registry.register(
            PythonREPLTool(timeout_seconds=python_timeout, memory_limit_mb=python_memory_mb)
        )
    register_if_enabled("datetime", DatetimeTool)
    register_if_enabled("unit_converter", UnitConverterTool)
    register_if_enabled("currency_converter", CurrencyConverterTool)
    register_if_enabled("weather", WeatherTool)
    register_if_enabled("file_reader", lambda: FileReaderTool(workspace_root=workspace_root))
    register_if_enabled("csv_analyzer", lambda: CSVAnalyzerTool(workspace_root=workspace_root))
    register_if_enabled("json_explorer", lambda: JSONExplorerTool(workspace_root=workspace_root))
    register_if_enabled("webpage_reader", WebpageReaderTool)
    register_if_enabled("markdown_reader", lambda: MarkdownReaderTool(workspace_root=workspace_root))
    register_if_enabled("document_search", lambda: DocumentSearchTool(workspace_root=workspace_root))
    register_if_enabled("local_rag", lambda: LocalRAGTool(workspace_root=workspace_root))

    if memory_store is not None:
        if is_enabled("semantic_search"):
            registry.register(SemanticSearchTool(store=memory_store))
        if is_enabled("vector_search"):
            registry.register(VectorSearchTool(store=memory_store))

    selected_optional = set(optional_tools or [])
    if "sqlite_query" in selected_optional and is_enabled("sqlite_query"):
        registry.register(SQLiteQueryTool(workspace_root=workspace_root))
    if "github_search" in selected_optional and is_enabled("github_search"):
        registry.register(GitHubSearchTool())
    if "news_search" in selected_optional and is_enabled("news_search"):
        registry.register(NewsSearchTool())
    if "arxiv_search" in selected_optional and is_enabled("arxiv_search"):
        registry.register(ArxivSearchTool())

    return registry
