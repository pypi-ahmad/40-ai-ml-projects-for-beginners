"""Factory methods for dynamic tool registration."""

from __future__ import annotations

from pathlib import Path

from crew_platform.memory.chroma_store import ChromaSemanticStore
from crew_platform.observability.metrics import MetricsStore
from crew_platform.observability.tracer import JsonlTracer
from crew_platform.tools.arxiv_search import ArxivSearchTool
from crew_platform.tools.calculator import CalculatorTool
from crew_platform.tools.chroma_retrieval import ChromaRetrievalTool
from crew_platform.tools.currency_converter import CurrencyConverterTool
from crew_platform.tools.csv_analyzer import CSVAnalyzerTool
from crew_platform.tools.csv_reader import CSVReaderTool
from crew_platform.tools.datetime_tool import DatetimeTool
from crew_platform.tools.document_search import DocumentSearchTool
from crew_platform.tools.file_reader import FileReaderTool
from crew_platform.tools.github_search import GitHubSearchTool
from crew_platform.tools.json_explorer import JSONExplorerTool
from crew_platform.tools.json_reader import JSONReaderTool
from crew_platform.tools.local_rag import LocalRAGTool
from crew_platform.tools.markdown_reader import MarkdownReaderTool
from crew_platform.tools.memory_search import MemorySearchTool
from crew_platform.tools.news_search import NewsSearchTool
from crew_platform.tools.pdf_reader import PDFReaderTool
from crew_platform.tools.python_repl import PythonREPLTool
from crew_platform.tools.registry import ToolRegistry
from crew_platform.tools.search import DuckDuckGoSearchTool
from crew_platform.tools.semantic_search import SemanticSearchTool
from crew_platform.tools.sql_query_tool import SQLQueryTool
from crew_platform.tools.sqlite_tool import SQLiteQueryTool
from crew_platform.tools.unit_converter import UnitConverterTool
from crew_platform.tools.url_fetcher import URLFetcherTool
from crew_platform.tools.vector_search import VectorSearchTool
from crew_platform.tools.weather import WeatherTool
from crew_platform.tools.web_search import WebSearchTool
from crew_platform.tools.webpage_reader import WebpageReaderTool
from crew_platform.tools.wikipedia_tool import WikipediaTool


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
    """Create and pre-register tools with alias compatibility."""

    registry = ToolRegistry(tracer=tracer, metrics=metrics)
    selected_required = set(enabled_tools or ["*"])

    def is_enabled(name: str) -> bool:
        return "*" in selected_required or name in selected_required

    def register_if_enabled(tool_name: str, factory) -> None:
        if is_enabled(tool_name):
            registry.register(factory())

    register_if_enabled("calculator", CalculatorTool)
    register_if_enabled("duckduckgo_search", DuckDuckGoSearchTool)
    register_if_enabled("web_search", WebSearchTool)
    register_if_enabled("wikipedia", WikipediaTool)
    register_if_enabled("datetime", DatetimeTool)
    register_if_enabled("unit_converter", UnitConverterTool)
    register_if_enabled("currency_converter", CurrencyConverterTool)
    register_if_enabled("weather", WeatherTool)

    register_if_enabled("file_reader", lambda: FileReaderTool(workspace_root=workspace_root))
    register_if_enabled("csv_analyzer", lambda: CSVAnalyzerTool(workspace_root=workspace_root))
    register_if_enabled("csv_reader", lambda: CSVReaderTool(workspace_root=workspace_root))
    register_if_enabled("json_explorer", lambda: JSONExplorerTool(workspace_root=workspace_root))
    register_if_enabled("json_reader", lambda: JSONReaderTool(workspace_root=workspace_root))
    register_if_enabled("markdown_reader", lambda: MarkdownReaderTool(workspace_root=workspace_root))
    register_if_enabled("pdf_reader", lambda: PDFReaderTool(workspace_root=workspace_root))
    register_if_enabled("webpage_reader", WebpageReaderTool)
    register_if_enabled("url_fetcher", URLFetcherTool)
    register_if_enabled("document_search", lambda: DocumentSearchTool(workspace_root=workspace_root))
    register_if_enabled("local_rag", lambda: LocalRAGTool(workspace_root=workspace_root))

    if enable_python_tool and is_enabled("python_repl"):
        registry.register(PythonREPLTool(timeout_seconds=python_timeout, memory_limit_mb=python_memory_mb))

    if memory_store is not None:
        if is_enabled("semantic_search"):
            registry.register(SemanticSearchTool(store=memory_store))
        if is_enabled("vector_search"):
            registry.register(VectorSearchTool(store=memory_store))
        if is_enabled("memory_search"):
            registry.register(MemorySearchTool(store=memory_store))
        if is_enabled("chroma_retrieval"):
            registry.register(ChromaRetrievalTool(store=memory_store))

    selected_optional = set(optional_tools or [])

    if "sqlite_query" in selected_optional and is_enabled("sqlite_query"):
        registry.register(SQLiteQueryTool(workspace_root=workspace_root))
    if "sql_query_tool" in selected_optional and is_enabled("sql_query_tool"):
        registry.register(SQLQueryTool(workspace_root=workspace_root))
    if "github_search" in selected_optional and is_enabled("github_search"):
        registry.register(GitHubSearchTool())
    if "news_search" in selected_optional and is_enabled("news_search"):
        registry.register(NewsSearchTool())
    if "arxiv_search" in selected_optional and is_enabled("arxiv_search"):
        registry.register(ArxivSearchTool())

    return registry
