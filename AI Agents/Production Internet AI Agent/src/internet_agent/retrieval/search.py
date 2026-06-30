"""Search providers for web/news/wiki/github."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus

import httpx
import wikipediaapi

from internet_agent.config import Settings

try:
    from ddgs import DDGS
except Exception:  # pragma: no cover
    from duckduckgo_search import DDGS


class SearchProviders:
    """Provider fan-out for internet search."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._wiki = wikipediaapi.Wikipedia(user_agent="internet-agent/0.1", language="en")
        self._last_net_check_at = 0.0
        self._network_available = True

    async def _check_network(self) -> bool:
        now = time.time()
        if now - self._last_net_check_at < 10:
            return self._network_available

        self._last_net_check_at = now
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get("https://duckduckgo.com")
            self._network_available = response.status_code < 500
        except Exception:
            self._network_available = False
        return self._network_available

    async def search_duckduckgo(self, query: str, max_results: int) -> list[dict[str, Any]]:
        def _run() -> list[dict[str, Any]]:
            with DDGS() as ddgs:
                rows = list(ddgs.text(query, max_results=max_results))
            return [
                {
                    "title": row.get("title", ""),
                    "url": row.get("href", ""),
                    "snippet": row.get("body", ""),
                    "source": "duckduckgo",
                    "published": row.get("date", ""),
                }
                for row in rows
                if row.get("href")
            ]

        return await asyncio.to_thread(_run)

    async def search_news(self, query: str, max_results: int) -> list[dict[str, Any]]:
        def _run() -> list[dict[str, Any]]:
            with DDGS() as ddgs:
                rows = list(ddgs.news(query, max_results=max_results))
            return [
                {
                    "title": row.get("title", ""),
                    "url": row.get("url", ""),
                    "snippet": row.get("body", ""),
                    "source": "news",
                    "published": row.get("date", ""),
                }
                for row in rows
                if row.get("url")
            ]

        return await asyncio.to_thread(_run)

    async def search_wikipedia(self, query: str, max_results: int) -> list[dict[str, Any]]:
        def _run() -> list[dict[str, Any]]:
            page = self._wiki.page(query)
            if not page.exists():
                return []
            return [
                {
                    "title": page.title,
                    "url": page.fullurl,
                    "snippet": page.summary[:400],
                    "source": "wikipedia",
                    "published": "",
                }
            ][:max_results]

        return await asyncio.to_thread(_run)

    async def search_github(self, query: str, max_results: int) -> list[dict[str, Any]]:
        url = f"https://api.github.com/search/repositories?q={quote_plus(query)}&per_page={max_results}"
        headers = {"Accept": "application/vnd.github+json"}
        async with httpx.AsyncClient(timeout=self.settings.search.request_timeout_seconds) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
        results = []
        for item in payload.get("items", []):
            results.append(
                {
                    "title": item.get("full_name", ""),
                    "url": item.get("html_url", ""),
                    "snippet": item.get("description", ""),
                    "source": "github",
                    "published": item.get("updated_at", ""),
                    "stars": item.get("stargazers_count", 0),
                }
            )
        return results

    async def search_all(
        self,
        query: str,
        providers: list[str] | None = None,
        max_results: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        selected = providers or self.settings.search.providers
        limit = max_results or self.settings.search.default_max_results

        if not await self._check_network():
            return {
                name: [
                    {
                        "title": f"{name} unavailable",
                        "url": "",
                        "snippet": "Network unavailable in current environment.",
                        "source": name,
                        "published": datetime.utcnow().isoformat(),
                        "error": True,
                    }
                ]
                for name in selected
            }

        tasks: dict[str, Any] = {}
        if "duckduckgo" in selected:
            tasks["duckduckgo"] = self.search_duckduckgo(query, limit)
        if "news" in selected:
            tasks["news"] = self.search_news(query, limit)
        if "wikipedia" in selected:
            tasks["wikipedia"] = self.search_wikipedia(query, min(1, limit))
        if "github" in selected:
            tasks["github"] = self.search_github(query, limit)

        if not tasks:
            return {}

        async def _run_with_budget(name: str, coro) -> list[dict[str, Any]]:
            try:
                return await asyncio.wait_for(
                    coro, timeout=self.settings.search.request_timeout_seconds
                )
            except Exception as exc:  # noqa: BLE001
                return [
                    {
                        "title": f"{name} error",
                        "url": "",
                        "snippet": str(exc),
                        "source": name,
                        "published": datetime.utcnow().isoformat(),
                        "error": True,
                    }
                ]

        names = list(tasks.keys())
        values = await asyncio.gather(
            *[_run_with_budget(name, tasks[name]) for name in names],
            return_exceptions=False,
        )
        out: dict[str, list[dict[str, Any]]] = {}
        for name, value in zip(names, values, strict=True):
            out[name] = value
        return out
