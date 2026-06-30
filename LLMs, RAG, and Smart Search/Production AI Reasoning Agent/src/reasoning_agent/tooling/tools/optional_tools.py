"""Optional tool placeholders for phase-2 extensions."""

from __future__ import annotations

from pydantic import BaseModel

from reasoning_agent.tooling.base import ToolContext, ToolSpec


class PlaceholderInput(BaseModel):
    """Generic placeholder input."""

    query: str = ""


class PlaceholderOutput(BaseModel):
    """Generic placeholder output."""

    available: bool
    message: str


def unavailable(_: PlaceholderInput, __: ToolContext) -> PlaceholderOutput:
    """Return unavailable marker for optional tools."""

    return PlaceholderOutput(
        available=False,
        message="Optional tool not enabled in this runtime. Enable adapter implementation to activate.",
    )


sqlite_spec = ToolSpec(
    name="sqlite",
    description="Optional SQLite query/search tool",
    input_model=PlaceholderInput,
    output_model=PlaceholderOutput,
    tags=["optional", "database"],
)

github_spec = ToolSpec(
    name="github_search",
    description="Optional GitHub search tool",
    input_model=PlaceholderInput,
    output_model=PlaceholderOutput,
    tags=["optional", "search"],
)

news_spec = ToolSpec(
    name="news_search",
    description="Optional news search tool",
    input_model=PlaceholderInput,
    output_model=PlaceholderOutput,
    tags=["optional", "search"],
    requires_network=True,
)

arxiv_spec = ToolSpec(
    name="arxiv_search",
    description="Optional arXiv paper search tool",
    input_model=PlaceholderInput,
    output_model=PlaceholderOutput,
    tags=["optional", "search", "research"],
    requires_network=True,
)
