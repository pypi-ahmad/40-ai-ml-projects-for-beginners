"""DuckDuckGo web search tool."""

from __future__ import annotations

from duckduckgo_search import DDGS
from pydantic import BaseModel, Field

from reasoning_agent.tooling.base import ToolContext, ToolSpec


class SearchInput(BaseModel):
    """Search input payload."""

    query: str = Field(min_length=2)
    max_results: int = Field(default=5, ge=1, le=15)


class SearchOutput(BaseModel):
    """Search output payload."""

    query: str
    results: list[dict[str, str]]


def search(payload: SearchInput, _: ToolContext) -> SearchOutput:
    """Search web and return normalized result set."""

    rows: list[dict[str, str]] = []
    with DDGS() as ddgs:
        for item in ddgs.text(payload.query, max_results=payload.max_results):
            rows.append(
                {
                    "title": str(item.get("title", "")),
                    "url": str(item.get("href", "")),
                    "snippet": str(item.get("body", "")),
                }
            )
    return SearchOutput(query=payload.query, results=rows)


spec = ToolSpec(
    name="duckduckgo_search",
    description="Web search using DuckDuckGo with snippets and links",
    input_model=SearchInput,
    output_model=SearchOutput,
    tags=["search", "web"],
    requires_network=True,
)
