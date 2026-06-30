"""Summarization module backed by local Ollama models."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.config import get_config
from src.ollama_client import OllamaClient
from src.schemas import ErrorInfo, SummaryResult

logger = logging.getLogger(__name__)


SUMMARY_SYSTEM_PROMPT = """You are expert summarizer.
Return only JSON with keys:
- summary: concise summary in 2-4 sentences
- key_points: list of 3-5 concise bullets
- original_length: integer word count estimate
No markdown, no extra text."""


class Summarizer:
    """Generate concise summaries with key points."""

    def __init__(
        self,
        model: str | None = None,
        client: OllamaClient | None = None,
        max_input_chars: int = 8_000,
    ) -> None:
        self.model = model or get_config().summarization_model
        self.client = client or OllamaClient()
        self.max_input_chars = max_input_chars
        self._owns_client = client is None

    def close(self) -> None:
        """Close owned client resources."""

        if self._owns_client:
            self.client.close()

    @staticmethod
    def _extract_json(raw_response: str) -> dict[str, Any] | None:
        """Extract JSON object from model response when possible."""

        left = raw_response.find("{")
        right = raw_response.rfind("}")
        if left == -1 or right == -1 or right <= left:
            return None

        try:
            return json.loads(raw_response[left : right + 1])
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_bullet_points(raw_response: str) -> list[str]:
        """Extract bullet-like lines when structured JSON parsing fails."""

        bullets: list[str] = []
        for line in raw_response.splitlines():
            stripped = line.strip()
            if stripped.startswith(("-", "*")):
                bullets.append(stripped.lstrip("-* ").strip())
        return bullets[:5]

    def summarize(self, text: str) -> SummaryResult:
        """Run full summarization pipeline for given text."""

        if not text or not text.strip():
            return SummaryResult(
                summary="",
                key_points=[],
                model=self.model,
                original_word_count=0,
                error=ErrorInfo(message="Input text cannot be empty.", stage="validation"),
            )

        normalized_text = " ".join(text.split())
        truncated_text = normalized_text[: self.max_input_chars]
        if len(normalized_text) > self.max_input_chars:
            logger.info(
                "Summarization input truncated from %s chars to %s chars",
                len(normalized_text),
                self.max_input_chars,
            )

        response = self.client.generate(
            model=self.model,
            prompt=truncated_text,
            system=SUMMARY_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=700,
        )

        if response["error"]:
            return SummaryResult(
                summary="",
                key_points=[],
                model=self.model,
                original_word_count=len(normalized_text.split()),
                latency_ms=response["latency_ms"],
                error=ErrorInfo(message=response["error"], stage="inference"),
            )

        parsed = self._extract_json(response["response"])
        if parsed:
            key_points = parsed.get("key_points")
            if not isinstance(key_points, list):
                key_points = []

            return SummaryResult(
                summary=str(parsed.get("summary", "")).strip() or response["response"][:500],
                key_points=[str(point) for point in key_points][:5],
                model=self.model,
                original_word_count=int(
                    parsed.get("original_length", len(normalized_text.split()))
                ),
                latency_ms=response["latency_ms"],
                error=None,
            )

        return SummaryResult(
            summary=response["response"][:700],
            key_points=self._extract_bullet_points(response["response"]),
            model=self.model,
            original_word_count=len(normalized_text.split()),
            latency_ms=response["latency_ms"],
            error=None,
        )


def summarize_text(
    text: str, model: str | None = None, client: OllamaClient | None = None
) -> dict[str, Any]:
    """Backwards-compatible wrapper returning dictionary payload."""

    summarizer = Summarizer(model=model, client=client)
    try:
        return summarizer.summarize(text).model_dump()
    finally:
        summarizer.close()
