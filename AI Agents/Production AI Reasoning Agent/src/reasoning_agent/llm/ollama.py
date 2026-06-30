"""Ollama provider client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


class OllamaError(RuntimeError):
    """Base Ollama provider error."""


@dataclass(slots=True)
class OllamaResponse:
    model: str
    text: str
    prompt_eval_count: int | None = None
    eval_count: int | None = None


class OllamaProvider:
    """Thin async client over Ollama HTTP API."""

    def __init__(self, base_url: str, timeout_seconds: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def healthcheck(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
            return True
        except Exception:  # noqa: BLE001
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4))
    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        raw: bool = False,
    ) -> OllamaResponse:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "raw": raw,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
        return OllamaResponse(
            model=model,
            text=str(data.get("response", "")).strip(),
            prompt_eval_count=data.get("prompt_eval_count"),
            eval_count=data.get("eval_count"),
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4))
    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> OllamaResponse:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        message = data.get("message", {})
        return OllamaResponse(
            model=model,
            text=str(message.get("content", "")).strip(),
            prompt_eval_count=data.get("prompt_eval_count"),
            eval_count=data.get("eval_count"),
        )

    async def embed(self, text: str, model: str = "qwen3-embedding:4b") -> list[float]:
        payload = {"model": model, "input": text}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/api/embed", json=payload)
            response.raise_for_status()
            data = response.json()
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise OllamaError("Embedding response missing vectors")
        return embeddings[0]

    async def list_models(self) -> set[str]:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
        return {item["name"] for item in data.get("models", []) if "name" in item}

    async def ensure_model_available(self, model: str, auto_pull: bool = True) -> bool:
        models = await self.list_models()
        if model in models:
            return True
        if not auto_pull:
            return False
        payload: dict[str, Any] = {"name": model, "stream": False}
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(f"{self.base_url}/api/pull", json=payload)
            response.raise_for_status()
        updated = await self.list_models()
        return model in updated
