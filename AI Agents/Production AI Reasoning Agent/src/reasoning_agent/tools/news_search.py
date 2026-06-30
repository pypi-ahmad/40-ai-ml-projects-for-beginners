"""Optional news search via DuckDuckGo news endpoint."""

from __future__ import annotations

from duckduckgo_search import DDGS
from pydantic import BaseModel

from reasoning_agent.tools.base import BaseTool
from reasoning_agent.utils.network import offline_mode


class NewsSearchInput(BaseModel):
    query: str
    max_results: int = 5


class NewsSearchOutput(BaseModel):
    results: list[dict[str, str]]


class NewsSearchTool(BaseTool[NewsSearchInput, NewsSearchOutput]):
    name = "news_search"
    description = "Searches latest news snippets"
    input_model = NewsSearchInput
    output_model = NewsSearchOutput

    async def run(self, payload: NewsSearchInput) -> NewsSearchOutput:
        results: list[dict[str, str]] = []
        if offline_mode():
            return NewsSearchOutput(
                results=[
                    {
                        "title": "offline-fallback",
                        "url": "",
                        "body": "news disabled by AGENT_OFFLINE_MODE",
                    }
                ]
            )
        try:
            with DDGS() as ddgs:
                for item in ddgs.news(payload.query, max_results=payload.max_results):
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "body": item.get("body", ""),
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            results.append({"title": "offline-fallback", "url": "", "body": f"news unavailable: {exc}"})
        return NewsSearchOutput(results=results)
