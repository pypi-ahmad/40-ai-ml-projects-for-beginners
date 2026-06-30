"""Built-in tool implementations for workflow agents."""

from __future__ import annotations

import ast
import io
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import httpx

from langgraph_platform.tools.base import ToolContext, ToolRegistry, ToolResult


def _ddgs_client():
    try:
        from ddgs import DDGS

        return DDGS(timeout=5)
    except Exception:
        from duckduckgo_search import DDGS

        return DDGS(timeout=5)


class DuckDuckGoSearchTool:
    name = "duckduckgo_search"

    def run(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, output=[], error="Missing query")
        try:
            with _ddgs_client() as ddgs:
                results = list(ddgs.text(query, max_results=int(args.get("max_results", 5))))
            return ToolResult(ok=True, output=results, source="duckduckgo")
        except Exception as exc:
            return ToolResult(ok=False, output=[], source="duckduckgo", error=str(exc))


class WikipediaTool:
    name = "wikipedia"

    def run(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, output=[], error="Missing query")
        try:
            import wikipedia

            page = wikipedia.page(query, auto_suggest=True)
            return ToolResult(
                ok=True,
                output={"title": page.title, "summary": page.summary, "url": page.url},
                source="wikipedia",
            )
        except Exception as exc:
            return ToolResult(ok=False, output={}, source="wikipedia", error=str(exc))


class GitHubSearchTool:
    name = "github_search"

    def run(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, output=[], error="Missing query")
        try:
            url = "https://api.github.com/search/repositories"
            response = httpx.get(
                url, params={"q": query, "per_page": int(args.get("max_results", 5))}, timeout=20
            )
            response.raise_for_status()
            payload = response.json()
            return ToolResult(ok=True, output=payload.get("items", []), source="github")
        except Exception as exc:
            return ToolResult(ok=False, output=[], source="github", error=str(exc))


class NewsSearchTool:
    name = "news_search"

    def run(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, output=[], error="Missing query")
        try:
            with _ddgs_client() as ddgs:
                results = list(ddgs.news(query, max_results=int(args.get("max_results", 5))))
            return ToolResult(ok=True, output=results, source="duckduckgo_news")
        except Exception as exc:
            return ToolResult(ok=False, output=[], source="duckduckgo_news", error=str(exc))


class WeatherTool:
    name = "weather"

    def run(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        location = str(args.get("location", "")).strip()
        if not location:
            return ToolResult(ok=False, output={}, error="Missing location")
        try:
            response = httpx.get(f"https://wttr.in/{location}", params={"format": "j1"}, timeout=20)
            response.raise_for_status()
            return ToolResult(ok=True, output=response.json(), source="wttr.in")
        except Exception as exc:
            return ToolResult(ok=False, output={}, source="wttr.in", error=str(exc))


class CalculatorTool:
    name = "calculator"

    _allowed_nodes = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Load,
        ast.Name,
    }
    _allowed_symbols = {"pi": math.pi, "e": math.e}

    def run(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        expression = str(args.get("expression", "")).strip()
        if not expression:
            return ToolResult(ok=False, output=None, error="Missing expression")
        try:
            tree = ast.parse(expression, mode="eval")
            for node in ast.walk(tree):
                if type(node) not in self._allowed_nodes:
                    return ToolResult(
                        ok=False, output=None, error=f"Unsupported token: {type(node).__name__}"
                    )
                if isinstance(node, ast.Name) and node.id not in self._allowed_symbols:
                    return ToolResult(ok=False, output=None, error=f"Unsupported symbol: {node.id}")
            result = eval(
                compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, self._allowed_symbols
            )
            return ToolResult(ok=True, output=result, source="calculator")
        except Exception as exc:
            return ToolResult(ok=False, output=None, error=str(exc))


class CurrencyConverterTool:
    name = "currency_converter"

    _fallback_rates = {"USD": 1.0, "EUR": 0.91, "GBP": 0.78, "INR": 83.6, "JPY": 159.0}

    def run(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        amount = float(args.get("amount", 0.0))
        from_currency = str(args.get("from", "USD")).upper()
        to_currency = str(args.get("to", "USD")).upper()
        if from_currency not in self._fallback_rates or to_currency not in self._fallback_rates:
            return ToolResult(ok=False, output=None, error="Unsupported currency")
        usd_amount = amount / self._fallback_rates[from_currency]
        converted = usd_amount * self._fallback_rates[to_currency]
        return ToolResult(
            ok=True, output={"amount": converted, "currency": to_currency}, source="fallback_rates"
        )


class UnitConverterTool:
    name = "unit_converter"

    _unit_factors = {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001, "mile": 1609.34, "ft": 0.3048}

    def run(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        value = float(args.get("value", 0.0))
        from_unit = str(args.get("from", "m"))
        to_unit = str(args.get("to", "m"))
        if from_unit not in self._unit_factors or to_unit not in self._unit_factors:
            return ToolResult(ok=False, output=None, error="Unsupported unit")
        base = value * self._unit_factors[from_unit]
        converted = base / self._unit_factors[to_unit]
        return ToolResult(
            ok=True, output={"value": converted, "unit": to_unit}, source="unit_converter"
        )


class PythonReplTool:
    name = "python_repl"

    def run(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        code = str(args.get("code", "")).strip()
        if not code:
            return ToolResult(ok=False, output="", error="Missing code")
        safe_globals = {
            "__builtins__": {"len": len, "range": range, "sum": sum, "min": min, "max": max}
        }
        safe_locals: dict[str, Any] = {}
        buffer = io.StringIO()
        try:
            exec(code, safe_globals, safe_locals)
            output = safe_locals.get("result", "")
            return ToolResult(ok=True, output=output, source="python_repl")
        except Exception as exc:
            return ToolResult(ok=False, output=buffer.getvalue(), error=str(exc))


class FileReaderTool:
    name = "file_reader"

    def run(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        path = Path(str(args.get("path", "")))
        if not path.exists():
            return ToolResult(ok=False, output="", error=f"File not found: {path}")
        return ToolResult(ok=True, output=path.read_text(encoding="utf-8"), source=str(path))


class MarkdownReaderTool(FileReaderTool):
    name = "markdown_reader"


class PDFReaderTool:
    name = "pdf_reader"

    def run(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        path = Path(str(args.get("path", "")))
        if not path.exists():
            return ToolResult(ok=False, output="", error=f"File not found: {path}")
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return ToolResult(ok=True, output=text, source=str(path))
        except Exception as exc:
            return ToolResult(ok=False, output="", source=str(path), error=str(exc))


class CSVReaderTool:
    name = "csv_reader"

    def run(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        path = Path(str(args.get("path", "")))
        if not path.exists():
            return ToolResult(ok=False, output=[], error=f"File not found: {path}")
        try:
            import pandas as pd

            frame = pd.read_csv(path)
            limit = int(args.get("limit", 50))
            return ToolResult(
                ok=True, output=frame.head(limit).to_dict(orient="records"), source=str(path)
            )
        except Exception as exc:
            return ToolResult(ok=False, output=[], source=str(path), error=str(exc))


class JSONReaderTool:
    name = "json_reader"

    def run(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        path = Path(str(args.get("path", "")))
        if not path.exists():
            return ToolResult(ok=False, output={}, error=f"File not found: {path}")
        return ToolResult(
            ok=True, output=json.loads(path.read_text(encoding="utf-8")), source=str(path)
        )


class SQLTool:
    name = "sql_tool"

    def run(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        db_path = Path(str(args.get("db_path", "artifacts/langgraph_platform.db")))
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, output=[], error="Missing query")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description or []]
        output = [dict(zip(columns, row, strict=False)) for row in rows]
        return ToolResult(ok=True, output=output, source=str(db_path))


class ChromaSearchTool:
    name = "chroma_search"

    def __init__(self, chroma_path: str = "artifacts/chroma") -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=chroma_path)
        self._collection = self._client.get_or_create_collection(name="knowledge")

    def run(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, output=[], error="Missing query")
        top_k = int(args.get("top_k", 5))
        result = self._collection.query(query_texts=[query], n_results=top_k)
        return ToolResult(ok=True, output=result, source="chroma")

    def close(self) -> None:
        """Close underlying Chroma persistent client."""

        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                return


class URLFetcherTool:
    name = "url_fetcher"

    def run(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        url = str(args.get("url", "")).strip()
        if not url:
            return ToolResult(ok=False, output="", error="Missing url")
        try:
            response = httpx.get(url, timeout=30)
            response.raise_for_status()
            raw_html = response.text
            try:
                import trafilatura

                extracted = trafilatura.extract(raw_html)
            except Exception:
                extracted = None
            if extracted:
                return ToolResult(ok=True, output=extracted, source=url)
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(raw_html, "html.parser")
            text = "\n".join(
                chunk.strip()
                for chunk in soup.get_text(separator="\n").splitlines()
                if chunk.strip()
            )
            return ToolResult(ok=True, output=text, source=url)
        except Exception as exc:
            return ToolResult(ok=False, output="", source=url, error=str(exc))


class MemorySearchTool:
    name = "memory_search"

    def __init__(self, db_path: str = "artifacts/langgraph_platform.db") -> None:
        self.db_path = Path(db_path)

    def run(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        query = str(args.get("query", "")).strip().lower()
        limit = int(args.get("limit", 5))
        if not self.db_path.exists():
            return ToolResult(ok=True, output=[], source=str(self.db_path))

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT workflow_id, user_request, final_report, created_at
                FROM workflow_runs
                WHERE lower(user_request) LIKE ? OR lower(final_report) LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()

        return ToolResult(ok=True, output=[dict(row) for row in rows], source=str(self.db_path))


class DocumentationSearchTool:
    name = "documentation_search"

    def run(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        path = Path(str(args.get("path", "docs")))
        query = str(args.get("query", "")).lower()
        matches: list[dict[str, Any]] = []
        for file in path.rglob("*.md"):
            text = file.read_text(encoding="utf-8")
            if query in text.lower():
                matches.append({"path": str(file), "snippet": text[:500]})
        return ToolResult(ok=True, output=matches[: int(args.get("limit", 5))], source=str(path))


class _NoOpChromaSearchTool:
    name = "chroma_search"

    def run(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        return ToolResult(
            ok=True,
            output={"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]},
            source="chroma_stub",
        )


def build_default_registry(
    chroma_path: str = "artifacts/chroma", db_path: str = "artifacts/langgraph_platform.db"
) -> ToolRegistry:
    """Create tool registry with all required built-ins."""

    registry = ToolRegistry()
    try:
        chroma_tool = ChromaSearchTool(chroma_path=chroma_path)
    except Exception:
        chroma_tool = _NoOpChromaSearchTool()

    tools = [
        DuckDuckGoSearchTool(),
        WikipediaTool(),
        GitHubSearchTool(),
        NewsSearchTool(),
        WeatherTool(),
        CalculatorTool(),
        CurrencyConverterTool(),
        UnitConverterTool(),
        PythonReplTool(),
        FileReaderTool(),
        MarkdownReaderTool(),
        PDFReaderTool(),
        CSVReaderTool(),
        JSONReaderTool(),
        SQLTool(),
        chroma_tool,
        URLFetcherTool(),
        MemorySearchTool(db_path=db_path),
        DocumentationSearchTool(),
    ]
    for tool in tools:
        registry.register(tool)

    return registry
