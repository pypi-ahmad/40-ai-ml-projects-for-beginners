"""Ollama-based generation and judging clients."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Iterator
from typing import Any

from ollama import Client


class OllamaLLM:
    """Ollama chat wrapper with sync and stream helpers."""

    def __init__(self, host: str, model: str, temperature: float, max_tokens: int) -> None:
        self.client = Client(host=host)
        self.host = host
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _options(self, temperature: float | None = None, max_tokens: int | None = None) -> dict[str, float | int]:
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
        started = time.perf_counter()
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options=self._options(temperature=temperature, max_tokens=max_tokens),
            think=False,
            stream=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        content = str(response.get("message", {}).get("content", "")).strip()
        return content, elapsed_ms

    async def async_generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, float]:
        return await asyncio.to_thread(
            self.generate,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        response_stream = self.client.chat(
            model=self.model,
            messages=messages,
            options=self._options(temperature=temperature, max_tokens=max_tokens),
            think=False,
            stream=True,
        )
        for chunk in response_stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token

    def close(self) -> None:
        """Close underlying HTTP client resources."""

        self.client.close()


class LLMJudge:
    """LLM-as-a-judge scoring for grounded response quality."""

    def __init__(self, host: str, model: str) -> None:
        self.client = Client(host=host)
        self.model = model

    def evaluate(self, query: str, answer: str, context: str) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        rubric_prompt = (
            "Evaluate answer quality using context only. Return JSON with keys: "
            "grounding, correctness, completeness, clarity, citation_quality, rationale. "
            "Scores must be integers 1-5.\n\n"
            f"Query: {query}\n"
            f"Answer: {answer}\n"
            f"Context: {context}\n"
        )
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": rubric_prompt}],
            options={"temperature": 0.0, "num_predict": 256},
            think=False,
        )
        payload = self._parse_json(str(response.get("message", {}).get("content", "")))
        elapsed_ms = (time.perf_counter() - started) * 1000
        return payload, elapsed_ms

    def close(self) -> None:
        """Close underlying HTTP client resources."""

        self.client.close()

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            payload = json.loads(match.group(0)) if match else {}
        out: dict[str, Any] = {
            "grounding": max(1, min(5, int(payload.get("grounding", 1)))),
            "correctness": max(1, min(5, int(payload.get("correctness", 1)))),
            "completeness": max(1, min(5, int(payload.get("completeness", 1)))),
            "clarity": max(1, min(5, int(payload.get("clarity", 1)))),
            "citation_quality": max(1, min(5, int(payload.get("citation_quality", 1)))),
            "rationale": str(payload.get("rationale", "")),
        }
        return out
