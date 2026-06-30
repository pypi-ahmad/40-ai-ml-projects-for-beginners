"""URL fetcher tool."""

from __future__ import annotations

import httpx
import trafilatura
from pydantic import BaseModel, HttpUrl

from crew_platform.tools.base import BaseTool
from crew_platform.utils.network import offline_mode


class URLFetcherInput(BaseModel):
    url: HttpUrl


class URLFetcherOutput(BaseModel):
    url: str
    title: str
    content: str


class URLFetcherTool(BaseTool[URLFetcherInput, URLFetcherOutput]):
    name = "url_fetcher"
    description = "Fetches and extracts article text from URL"
    input_model = URLFetcherInput
    output_model = URLFetcherOutput

    async def run(self, payload: URLFetcherInput) -> URLFetcherOutput:
        if offline_mode():
            return URLFetcherOutput(url=str(payload.url), title="offline-fallback", content="network disabled")

        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(str(payload.url), follow_redirects=True)
                response.raise_for_status()
            extracted = trafilatura.extract(response.text, include_links=True, include_comments=False)
            content = extracted or response.text[:5000]
            title = str(payload.url)
            return URLFetcherOutput(url=str(payload.url), title=title, content=content[:25000])
        except Exception as exc:  # noqa: BLE001
            return URLFetcherOutput(url=str(payload.url), title="unavailable", content=f"fetch failed: {exc}")
