"""Wikipedia lookup tool."""

from __future__ import annotations

import wikipediaapi
from pydantic import BaseModel

from reasoning_agent.tools.base import BaseTool
from reasoning_agent.utils.network import offline_mode


class WikipediaInput(BaseModel):
    query: str


class WikipediaOutput(BaseModel):
    title: str
    summary: str
    url: str


class WikipediaTool(BaseTool[WikipediaInput, WikipediaOutput]):
    name = "wikipedia"
    description = "Searches Wikipedia and returns summary"
    input_model = WikipediaInput
    output_model = WikipediaOutput

    def __init__(self) -> None:
        self._client = wikipediaapi.Wikipedia(user_agent="production-reasoning-agent/0.1", language="en")

    async def run(self, payload: WikipediaInput) -> WikipediaOutput:
        if offline_mode():
            return WikipediaOutput(
                title="offline-fallback",
                summary="wikipedia disabled by AGENT_OFFLINE_MODE",
                url="",
            )
        try:
            page = self._client.page(payload.query)
            if not page.exists():
                raise ValueError(f"Wikipedia page not found: {payload.query}")
            return WikipediaOutput(title=page.title, summary=page.summary[:2000], url=page.fullurl)
        except Exception as exc:  # noqa: BLE001
            return WikipediaOutput(
                title="offline-fallback",
                summary=f"wikipedia unavailable: {exc}",
                url="",
            )
