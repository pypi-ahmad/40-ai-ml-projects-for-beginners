"""HTTP fetchers for website and document retrieval."""

from __future__ import annotations

import time
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from internet_agent.metrics import METRICS


class WebsiteFetcher:
    """Robust async website fetcher with retries and timeout."""

    def __init__(self, timeout_seconds: int = 20) -> None:
        self.timeout_seconds = timeout_seconds

    @retry(
        wait=wait_exponential(multiplier=0.25, min=0.5, max=2),
        stop=stop_after_attempt(2),
        reraise=True,
    )
    async def fetch(self, url: str) -> dict[str, Any]:
        started = time.perf_counter()
        timeout = httpx.Timeout(self.timeout_seconds, connect=min(5.0, self.timeout_seconds))
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = {
                "url": str(response.url),
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "text": response.text,
                "bytes": response.content,
            }
        METRICS.observe_ms("http.fetch.latency_ms", (time.perf_counter() - started) * 1000)
        return payload
