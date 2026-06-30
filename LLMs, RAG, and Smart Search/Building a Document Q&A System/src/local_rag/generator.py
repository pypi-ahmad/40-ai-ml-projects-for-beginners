"""Ollama generation wrapper with streaming support."""

from __future__ import annotations

import time
from collections.abc import Iterator

from ollama import Client


class OllamaGenerator:
    """Generate grounded answers from Ollama chat endpoint."""

    def __init__(self, model: str, host: str, temperature: float, max_tokens: int) -> None:
        self.model = model
        self.client = Client(host=host)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _options(
        self,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, float | int]:
        return {
            "temperature": self.temperature if temperature is None else temperature,
            "num_predict": self.max_tokens if max_tokens is None else max_tokens,
        }

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, float]:
        """Generate full response and return answer plus generation latency."""

        options = self._options(temperature=temperature, max_tokens=max_tokens)

        started = time.perf_counter()
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options=options,
            think=False,
            stream=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        answer = response.get("message", {}).get("content", "")
        return answer, elapsed_ms

    def stream_generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Stream generation tokens from Ollama chat endpoint."""

        options = self._options(temperature=temperature, max_tokens=max_tokens)
        response_stream = self.client.chat(
            model=self.model,
            messages=messages,
            options=options,
            think=False,
            stream=True,
        )
        for chunk in response_stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token
