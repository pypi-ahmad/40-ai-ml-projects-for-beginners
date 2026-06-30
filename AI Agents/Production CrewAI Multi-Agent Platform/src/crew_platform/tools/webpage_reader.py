"""Webpage reader tool."""

from __future__ import annotations

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, HttpUrl

from crew_platform.tools.base import BaseTool
from crew_platform.utils.network import offline_mode


class WebpageReaderInput(BaseModel):
    url: HttpUrl


class WebpageReaderOutput(BaseModel):
    title: str
    text: str


class WebpageReaderTool(BaseTool[WebpageReaderInput, WebpageReaderOutput]):
    name = "webpage_reader"
    description = "Fetch and summarize webpage content"
    input_model = WebpageReaderInput
    output_model = WebpageReaderOutput

    async def run(self, payload: WebpageReaderInput) -> WebpageReaderOutput:
        if offline_mode():
            return WebpageReaderOutput(title="offline-fallback", text="web disabled by AGENT_OFFLINE_MODE")
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(str(payload.url))
                response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled"
            text = " ".join(soup.get_text(separator=" ").split())
            return WebpageReaderOutput(title=title, text=text[:5000])
        except Exception as exc:  # noqa: BLE001
            return WebpageReaderOutput(title="offline-fallback", text=f"webpage unavailable: {exc}")
