"""Async Ollama client wrapper with retries and structured helpers."""

from __future__ import annotations

import time
from typing import Any

import httpx
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from api_intel_agent.config import load_settings


class OllamaProvider:
    def __init__(self, base_url: str | None = None, timeout_seconds: int | None = None) -> None:
        settings = load_settings()
        self.base_url = base_url or settings.llm.base_url
        self.timeout_seconds = timeout_seconds or settings.llm.timeout_seconds

    async def _request(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8)
        ):
            with attempt:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    started = time.perf_counter()
                    response = await client.post(f"{self.base_url}{endpoint}", json=payload)
                    response.raise_for_status()
                    body = response.json()
                    body["_latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
                    return body
        return {"error": "unknown"}

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        settings = load_settings()
        payload = {
            "model": model or settings.llm.default_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else settings.llm.temperature,
                "num_predict": max_tokens if max_tokens is not None else settings.llm.max_tokens,
            },
        }
        return await self._request("/api/generate", payload)

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        settings = load_settings()
        payload = {
            "model": model or settings.llm.default_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else settings.llm.temperature,
                "num_predict": settings.llm.max_tokens,
            },
        }
        return await self._request("/api/chat", payload)

    async def embed(self, text: str, model: str = "nomic-embed-text") -> list[float]:
        payload = {"model": model, "input": text}
        body = await self._request("/api/embed", payload)
        embeddings = body.get("embeddings") or []
        if embeddings and isinstance(embeddings[0], list):
            return embeddings[0]
        return embeddings

    async def healthcheck(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/ps")
                return response.status_code == 200
        except Exception:
            return False
