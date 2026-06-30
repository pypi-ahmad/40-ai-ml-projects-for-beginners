"""Web search providers with retries, caching, and provider switching."""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from ddgs import DDGS
from tenacity import retry, stop_after_attempt, wait_exponential

from hybrid_research_assistant.utils import json_dump, json_load, sha256_text


@dataclass(slots=True)
class WebResult:
    """Normalized web search result."""

    title: str
    url: str
    snippet: str
    score: float
    provider: str


class SearchProvider(Protocol):
    """Web search provider interface."""

    async def search(self, query: str, k: int, freshness_days: int) -> list[WebResult]:
        """Run web search."""


class CacheStore:
    """Simple JSON file cache for web responses."""

    def __init__(self, cache_dir: Path, ttl_seconds: int) -> None:
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> list[dict[str, Any]] | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        payload = json_load(path)
        expires_at = float(payload.get("expires_at", 0.0))
        if time.time() > expires_at:
            return None
        return list(payload.get("rows", []))

    def put(self, key: str, rows: list[dict[str, Any]]) -> None:
        payload = {
            "expires_at": time.time() + self.ttl_seconds,
            "rows": rows,
        }
        json_dump(self.cache_dir / f"{key}.json", payload)


class DuckDuckGoProvider:
    """DuckDuckGo provider (default, no API key)."""

    def __init__(self, timeout_seconds: int = 15) -> None:
        self.timeout_seconds = timeout_seconds

    async def search(self, query: str, k: int, freshness_days: int) -> list[WebResult]:
        def _run() -> list[WebResult]:
            ddgs = DDGS()
            rows = list(ddgs.text(query, max_results=k))
            results: list[WebResult] = []
            for idx, row in enumerate(rows, start=1):
                results.append(
                    WebResult(
                        title=str(row.get("title", "")),
                        url=str(row.get("href", "")),
                        snippet=str(row.get("body", "")),
                        score=max(0.0, 1.0 - ((idx - 1) * 0.1)),
                        provider="duckduckgo",
                    )
                )
            return results

        try:
            return await asyncio.wait_for(asyncio.to_thread(_run), timeout=self.timeout_seconds)
        except Exception:  # noqa: BLE001
            return []


class TavilyProvider:
    """Tavily provider adapter."""

    def __init__(self, api_key: str, timeout_seconds: int = 15) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4), reraise=True)
    async def _call(self, query: str, k: int) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={"query": query, "max_results": k, "api_key": self.api_key},
            )
            response.raise_for_status()
            return response.json()

    async def search(self, query: str, k: int, freshness_days: int) -> list[WebResult]:
        if not self.api_key:
            return []
        payload = await self._call(query, k)
        rows = payload.get("results", [])
        return [
            WebResult(
                title=str(row.get("title", "")),
                url=str(row.get("url", "")),
                snippet=str(row.get("content", "")),
                score=float(row.get("score", 0.0)),
                provider="tavily",
            )
            for row in rows
        ]


class BraveProvider:
    """Brave search provider adapter."""

    def __init__(self, api_key: str, timeout_seconds: int = 15) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4), reraise=True)
    async def _call(self, query: str, k: int) -> dict[str, Any]:
        headers = {"X-Subscription-Token": self.api_key, "Accept": "application/json"}
        params = {"q": query, "count": k}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
            response.raise_for_status()
            return response.json()

    async def search(self, query: str, k: int, freshness_days: int) -> list[WebResult]:
        if not self.api_key:
            return []
        payload = await self._call(query, k)
        rows = payload.get("web", {}).get("results", [])
        results: list[WebResult] = []
        for idx, row in enumerate(rows, start=1):
            results.append(
                WebResult(
                    title=str(row.get("title", "")),
                    url=str(row.get("url", "")),
                    snippet=str(row.get("description", "")),
                    score=max(0.0, 1.0 - ((idx - 1) * 0.1)),
                    provider="brave",
                )
            )
        return results


class WebSearchService:
    """Provider router with cache, retries, and filtering."""

    def __init__(
        self,
        *,
        default_provider: str,
        cache_store: CacheStore,
        providers: dict[str, SearchProvider],
    ) -> None:
        self.default_provider = default_provider
        self.cache_store = cache_store
        self.providers = providers

    async def search(
        self,
        query: str,
        *,
        k: int,
        provider: str | None = None,
        freshness_days: int = 7,
        allow_domains: list[str] | None = None,
    ) -> list[WebResult]:
        selected = provider or self.default_provider
        cache_key = sha256_text(f"{selected}|{query}|{k}|{freshness_days}|{allow_domains or []}")
        cached = self.cache_store.get(cache_key)
        if cached is not None:
            return [WebResult(**row) for row in cached]

        backend = self.providers.get(selected)
        if backend is None:
            return []

        rows = await backend.search(query=query, k=k, freshness_days=freshness_days)
        filtered = self._filter_domains(rows, allow_domains)
        self.cache_store.put(cache_key, [asdict(row) for row in filtered])
        return filtered

    @staticmethod
    def _filter_domains(rows: list[WebResult], allow_domains: list[str] | None) -> list[WebResult]:
        if not allow_domains:
            return rows
        allowed = tuple(domain.lower() for domain in allow_domains)
        return [row for row in rows if row.url.lower().find(allowed) != -1]
