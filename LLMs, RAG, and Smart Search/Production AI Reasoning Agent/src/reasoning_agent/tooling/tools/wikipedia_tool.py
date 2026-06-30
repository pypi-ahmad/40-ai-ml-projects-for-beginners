"""Wikipedia lookup tool."""

from __future__ import annotations

import wikipedia
from pydantic import BaseModel, Field

from reasoning_agent.tooling.base import ToolContext, ToolSpec


class WikipediaInput(BaseModel):
    """Wikipedia input payload."""

    query: str = Field(min_length=2)
    sentences: int = Field(default=3, ge=1, le=8)


class WikipediaOutput(BaseModel):
    """Wikipedia output payload."""

    query: str
    title: str
    summary: str
    url: str


def wikipedia_lookup(payload: WikipediaInput, _: ToolContext) -> WikipediaOutput:
    """Fetch short summary and citation URL from Wikipedia."""

    title = wikipedia.search(payload.query, results=1)
    if not title:
        raise ValueError("No Wikipedia page found")
    page = wikipedia.page(title[0], auto_suggest=False)
    summary = wikipedia.summary(title[0], sentences=payload.sentences, auto_suggest=False)
    return WikipediaOutput(query=payload.query, title=page.title, summary=summary, url=page.url)


spec = ToolSpec(
    name="wikipedia",
    description="Retrieve concise topic summary from Wikipedia",
    input_model=WikipediaInput,
    output_model=WikipediaOutput,
    tags=["search", "knowledge"],
    requires_network=True,
)
