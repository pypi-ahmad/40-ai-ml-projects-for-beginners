"""Search and content extraction tools."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import wikipediaapi
from pydantic import BaseModel

from internet_agent.retrieval.extract import clean_html, extract_markdown, read_pdf_bytes
from internet_agent.tools.base import BaseTool

try:
    from ddgs import DDGS
except Exception:  # pragma: no cover
    from duckduckgo_search import DDGS


class QueryInput(BaseModel):
    query: str
    max_results: int = 5


class SearchOutput(BaseModel):
    results: list[dict[str, Any]]


class DuckDuckGoSearchTool(BaseTool[QueryInput, SearchOutput]):
    name = "duckduckgo_search"
    description = "Search web pages with DuckDuckGo."
    input_model = QueryInput
    output_model = SearchOutput

    async def run(self, payload: QueryInput) -> SearchOutput:
        def _run() -> list[dict[str, Any]]:
            with DDGS() as ddgs:
                rows = list(ddgs.text(payload.query, max_results=payload.max_results))
            return [
                {
                    "title": row.get("title", ""),
                    "url": row.get("href", ""),
                    "snippet": row.get("body", ""),
                }
                for row in rows
                if row.get("href")
            ]

        rows = await asyncio.to_thread(_run)
        return SearchOutput(results=rows)


class UrlInput(BaseModel):
    url: str


class ContentOutput(BaseModel):
    url: str
    content: str


class WebsiteFetcherTool(BaseTool[UrlInput, ContentOutput]):
    name = "website_fetcher"
    description = "Fetch raw website content by URL."
    input_model = UrlInput
    output_model = ContentOutput

    async def run(self, payload: UrlInput) -> ContentOutput:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            response = await client.get(payload.url)
            response.raise_for_status()
        return ContentOutput(url=str(response.url), content=response.text)


class HtmlCleanerTool(BaseTool[ContentOutput, ContentOutput]):
    name = "html_cleaner"
    description = "Clean HTML and extract plain readable text."
    input_model = ContentOutput
    output_model = ContentOutput

    async def run(self, payload: ContentOutput) -> ContentOutput:
        return ContentOutput(url=payload.url, content=clean_html(payload.content))


class MarkdownExtractorTool(BaseTool[ContentOutput, ContentOutput]):
    name = "markdown_extractor"
    description = "Extract markdown from HTML content."
    input_model = ContentOutput
    output_model = ContentOutput

    async def run(self, payload: ContentOutput) -> ContentOutput:
        return ContentOutput(url=payload.url, content=extract_markdown(payload.content))


class PDFInput(BaseModel):
    path_or_url: str


class PDFOutput(BaseModel):
    content: str


class PDFReaderTool(BaseTool[PDFInput, PDFOutput]):
    name = "pdf_reader"
    description = "Read PDF from local path or URL."
    input_model = PDFInput
    output_model = PDFOutput

    async def run(self, payload: PDFInput) -> PDFOutput:
        if payload.path_or_url.startswith("http"):
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                response = await client.get(payload.path_or_url)
                response.raise_for_status()
                data = response.content
        else:
            with open(payload.path_or_url, "rb") as file_handle:
                data = file_handle.read()
        return PDFOutput(content=read_pdf_bytes(data))


class WikipediaSearchTool(BaseTool[QueryInput, SearchOutput]):
    name = "wikipedia_search"
    description = "Search Wikipedia summary for a topic."
    input_model = QueryInput
    output_model = SearchOutput

    def __init__(self) -> None:
        self._wiki = wikipediaapi.Wikipedia(user_agent="internet-agent/0.1", language="en")

    async def run(self, payload: QueryInput) -> SearchOutput:
        def _run() -> list[dict[str, Any]]:
            page = self._wiki.page(payload.query)
            if not page.exists():
                return []
            return [
                {
                    "title": page.title,
                    "url": page.fullurl,
                    "snippet": page.summary[:500],
                }
            ]

        rows = await asyncio.to_thread(_run)
        return SearchOutput(results=rows)


class GitHubSearchTool(BaseTool[QueryInput, SearchOutput]):
    name = "github_search"
    description = "Search GitHub repositories via public REST API."
    input_model = QueryInput
    output_model = SearchOutput

    async def run(self, payload: QueryInput) -> SearchOutput:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                "https://api.github.com/search/repositories",
                params={"q": payload.query, "per_page": payload.max_results},
                headers={"Accept": "application/vnd.github+json"},
            )
            response.raise_for_status()
            data = response.json()

        rows = [
            {
                "title": item.get("full_name", ""),
                "url": item.get("html_url", ""),
                "snippet": item.get("description", ""),
                "stars": item.get("stargazers_count", 0),
            }
            for item in data.get("items", [])
        ]
        return SearchOutput(results=rows)


class NewsSearchTool(BaseTool[QueryInput, SearchOutput]):
    name = "news_search"
    description = "Search current news with DuckDuckGo news index."
    input_model = QueryInput
    output_model = SearchOutput

    async def run(self, payload: QueryInput) -> SearchOutput:
        def _run() -> list[dict[str, Any]]:
            with DDGS() as ddgs:
                rows = list(ddgs.news(payload.query, max_results=payload.max_results))
            return [
                {
                    "title": row.get("title", ""),
                    "url": row.get("url", ""),
                    "snippet": row.get("body", ""),
                    "published": row.get("date", ""),
                }
                for row in rows
            ]

        rows = await asyncio.to_thread(_run)
        return SearchOutput(results=rows)
