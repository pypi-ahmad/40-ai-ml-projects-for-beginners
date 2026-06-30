"""Ollama clients for deterministic generation and structured responses."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_ollama import ChatOllama

from ai_sql_assistant.config import AppSettings
from ai_sql_assistant.logging_utils import logger


@dataclass(slots=True)
class LLMResponse:
    """Unified LLM response payload."""

    text: str
    latency_ms: float
    raw: dict[str, Any]


class OllamaDeterministicClient:
    """Direct Ollama REST client with deterministic decoding defaults."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.base_url = str(settings.ollama_host).rstrip("/")
        self._client = httpx.Client(timeout=settings.models.timeout_seconds)

    def close(self) -> None:
        """Close underlying HTTP client."""
        self._client.close()

    def _post_generate_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /api/generate with retry on transient Ollama/network issues."""
        retries = self.settings.models.request_retries
        backoff = self.settings.models.retry_backoff_seconds

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = self._client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                return response.json()
            except (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt >= retries:
                    break
                sleep_for = backoff * (attempt + 1)
                logger.warning(
                    "Ollama generate failed (attempt {}/{}): {}. Retrying in {:.1f}s",
                    attempt + 1,
                    retries + 1,
                    exc,
                    sleep_for,
                )
                time.sleep(sleep_for)

        assert last_error is not None
        raise last_error

    def generate(self, prompt: str, model: str) -> LLMResponse:
        """Generate non-streaming response from Ollama /api/generate."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.settings.models.temperature,
                "top_p": self.settings.models.top_p,
                "seed": 7,
                "num_predict": self.settings.models.num_predict,
            },
        }
        start = time.perf_counter()
        body = self._post_generate_with_retries(payload)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return LLMResponse(
            text=str(body.get("response", "")).strip(),
            latency_ms=elapsed_ms,
            raw=body,
        )

    def generate_json(self, prompt: str, model: str) -> LLMResponse:
        """Generate JSON-mode response for judge scoring."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.settings.models.temperature,
                "top_p": self.settings.models.top_p,
                "seed": 7,
                "num_predict": self.settings.models.num_predict,
            },
        }
        start = time.perf_counter()
        body = self._post_generate_with_retries(payload)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _ = json.loads(str(body.get("response", "{}")))
        return LLMResponse(
            text=str(body.get("response", "")).strip(),
            latency_ms=elapsed_ms,
            raw=body,
        )


class LangChainOllamaFactory:
    """Factory for LangChain ChatOllama instances."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def chat_model(self, model: str) -> ChatOllama:
        """Build ChatOllama with deterministic parameters."""
        return ChatOllama(
            model=model,
            base_url=str(self.settings.ollama_host).rstrip("/"),
            temperature=self.settings.models.temperature,
            top_p=self.settings.models.top_p,
            seed=7,
            num_ctx=8192,
            num_predict=self.settings.models.num_predict,
            disable_streaming=True,
            client_kwargs={"timeout": self.settings.models.timeout_seconds},
        )
