"""DuckDuckGo web search tool."""

from __future__ import annotations

from duckduckgo_search import DDGS
from pydantic import BaseModel

from reasoning_agent.tools.base import BaseTool
from reasoning_agent.utils.network import offline_mode


class SearchInput(BaseModel):
    query: str
    max_results: int = 5


class SearchOutput(BaseModel):
    query: str
    results: list[dict[str, str]]


class DuckDuckGoSearchTool(BaseTool[SearchInput, SearchOutput]):
    name = "duckduckgo_search"
    description = "Performs DuckDuckGo web search"
    input_model = SearchInput
    output_model = SearchOutput

    async def run(self, payload: SearchInput) -> SearchOutput:
        results: list[dict[str, str]] = []
        if offline_mode():
            results.append(
                {
                    "title": "offline-fallback",
                    "href": "",
                    "body": "search disabled by AGENT_OFFLINE_MODE",
                }
            )
            return SearchOutput(query=payload.query, results=results)
        try:
            with DDGS() as ddgs:
                for item in ddgs.text(payload.query, max_results=payload.max_results):
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "href": item.get("href", ""),
                            "body": item.get("body", ""),
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "title": "offline-fallback",
                    "href": "",
                    "body": f"search unavailable: {exc}",
                }
            )
        return SearchOutput(query=payload.query, results=results)
