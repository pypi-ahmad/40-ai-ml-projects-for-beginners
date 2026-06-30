"""Ollama API client with timing and basic retry support."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class LLMResult:
    """Unified model output."""

    text: str
    latency_ms: float
    model: str


class OllamaClient:
    """HTTP client for local Ollama models."""

    def __init__(self, base_url: str, timeout_seconds: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout_seconds)

    @staticmethod
    def _raise_for_status_with_body(resp: httpx.Response, endpoint: str) -> None:
        """Raise HTTPStatusError with trimmed response body context."""

        if resp.status_code < 400:
            return
        body = resp.text[:500]
        message = f"Ollama {endpoint} failed ({resp.status_code}): {body}"
        raise httpx.HTTPStatusError(message, request=resp.request, response=resp)

    def close(self) -> None:
        """Close HTTP session."""

        self._client.close()

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1200,
        raw: bool = True,
    ) -> LLMResult:
        """Call `/api/generate` non-stream mode."""

        started = time.perf_counter()
        resp = self._client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                "raw": raw,
            },
        )
        self._raise_for_status_with_body(resp, "/api/generate")
        payload = resp.json()
        latency_ms = (time.perf_counter() - started) * 1000
        text = str(payload.get("response", "")).strip()
        return LLMResult(text=text, latency_ms=latency_ms, model=model)

    def chat(self, model: str, messages: list[dict[str, str]], *, temperature: float = 0.2) -> LLMResult:
        """Call `/api/chat` non-stream mode."""

        started = time.perf_counter()
        resp = self._client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        self._raise_for_status_with_body(resp, "/api/chat")
        payload = resp.json()
        latency_ms = (time.perf_counter() - started) * 1000
        text = str(payload.get("message", {}).get("content", "")).strip()
        return LLMResult(text=text, latency_ms=latency_ms, model=model)

    def embed(self, model: str, text: str) -> list[float]:
        """Call `/api/embed` and return embedding vector."""

        resp = self._client.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": text},
        )
        self._raise_for_status_with_body(resp, "/api/embed")
        payload = resp.json()
        vectors = payload.get("embeddings") or payload.get("embedding") or []
        if isinstance(vectors, list) and vectors and isinstance(vectors[0], list):
            return [float(x) for x in vectors[0]]
        if isinstance(vectors, list):
            return [float(x) for x in vectors]
        return []

    def available_models(self) -> list[str]:
        """List local models from `/api/tags`."""

        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            return []
        models = payload.get("models", [])
        names: list[str] = []
        for item in models:
            name = item.get("name")
            if isinstance(name, str):
                names.append(name)
        return names

    def is_model_available(self, model: str) -> bool:
        """Check if model exists locally."""

        return model in self.available_models()

    def health(self) -> dict[str, Any]:
        """Health check for Ollama server."""

        try:
            response = self._client.get(f"{self.base_url}/")
            return {
                "ok": response.status_code < 500,
                "status": response.status_code,
                "text": response.text[:200],
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
