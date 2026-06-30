"""Webpage reader tool."""

from __future__ import annotations

import re

import httpx
from pydantic import BaseModel, HttpUrl

from reasoning_agent.tooling.base import ToolContext, ToolSpec


class WebpageReaderInput(BaseModel):
    """Webpage reader input payload."""

    url: HttpUrl
    max_chars: int = 5000


class WebpageReaderOutput(BaseModel):
    """Webpage reader output payload."""

    url: str
    content: str


def _strip_html(raw: str) -> str:
    text = re.sub(r"<script[\\s\\S]*?</script>", "", raw, flags=re.IGNORECASE)
    text = re.sub(r"<style[\\s\\S]*?</style>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def read_webpage(payload: WebpageReaderInput, _: ToolContext) -> WebpageReaderOutput:
    """Fetch webpage and return plain text."""

    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        resp = client.get(str(payload.url))
        resp.raise_for_status()
    text = _strip_html(resp.text)
    return WebpageReaderOutput(url=str(payload.url), content=text[: payload.max_chars])


spec = ToolSpec(
    name="webpage_reader",
    description="Fetch URL and extract plain text content",
    input_model=WebpageReaderInput,
    output_model=WebpageReaderOutput,
    tags=["web", "search"],
    requires_network=True,
)
