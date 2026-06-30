"""Sentiment analysis module backed by local Ollama inference."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.config import get_config
from src.ollama_client import OllamaClient
from src.schemas import ErrorInfo, SentimentResult

logger = logging.getLogger(__name__)


SENTIMENT_SYSTEM_PROMPT = """You are sentiment classifier.
Return only JSON with keys:
- label: Positive | Negative | Neutral
- confidence: float [0.0, 1.0]
- explanation: short sentence
No markdown, no extra text."""


class SentimentAnalyzer:
    """Classify sentiment and return normalized typed output."""

    def __init__(self, model: str | None = None, client: OllamaClient | None = None) -> None:
        self.model = model or get_config().sentiment_model
        self.client = client or OllamaClient()
        self._owns_client = client is None

    def close(self) -> None:
        """Close owned client resources."""

        if self._owns_client:
            self.client.close()

    def _parse_model_output(self, raw_response: str) -> tuple[str, float, str]:
        """Parse JSON first, then fall back to robust keyword heuristic."""

        json_candidate = re.search(r"\{[\s\S]*?\}", raw_response, re.DOTALL)
        if json_candidate:
            try:
                data = json.loads(json_candidate.group(0))
                label = str(data.get("label", "Neutral")).capitalize()
                confidence = float(data.get("confidence", 0.5))
                explanation = str(data.get("explanation", "No explanation provided."))
                if label not in {"Positive", "Negative", "Neutral"}:
                    label = "Neutral"
                confidence = min(max(confidence, 0.0), 1.0)
                return label, confidence, explanation
            except (ValueError, TypeError, json.JSONDecodeError):
                logger.warning("Structured sentiment parse failed. Falling back to heuristic.")

        lowered = raw_response.lower()
        positive_tokens = ["excellent", "good", "great", "happy", "amazing", "love", "positive"]
        negative_tokens = ["bad", "poor", "awful", "terrible", "hate", "negative", "sad"]

        positive_hits = sum(token in lowered for token in positive_tokens)
        negative_hits = sum(token in lowered for token in negative_tokens)

        if positive_hits > negative_hits:
            return "Positive", 0.62, raw_response[:200]
        if negative_hits > positive_hits:
            return "Negative", 0.62, raw_response[:200]
        return "Neutral", 0.5, raw_response[:200]

    def analyze(self, text: str) -> SentimentResult:
        """Run full sentiment pipeline: validation -> inference -> parsing."""

        if not text or not text.strip():
            return SentimentResult(
                label="Error",
                confidence=0.0,
                explanation="Input text cannot be empty.",
                model=self.model,
                error=ErrorInfo(message="Input text cannot be empty.", stage="validation"),
            )

        response = self.client.generate(
            model=self.model,
            prompt=text.strip(),
            system=SENTIMENT_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=160,
        )

        if response["error"]:
            return SentimentResult(
                label="Error",
                confidence=0.0,
                explanation="Sentiment request failed.",
                model=self.model,
                latency_ms=response["latency_ms"],
                error=ErrorInfo(message=response["error"], stage="inference"),
            )

        label, confidence, explanation = self._parse_model_output(response["response"])
        return SentimentResult(
            label=label,
            confidence=confidence,
            explanation=explanation,
            model=self.model,
            latency_ms=response["latency_ms"],
            error=None,
        )

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Run sentiment analysis for every text payload."""

        return [self.analyze(text) for text in texts]


def analyze_sentiment(
    text: str, model: str | None = None, client: OllamaClient | None = None
) -> dict[str, Any]:
    """Backwards-compatible function wrapper returning plain dictionary."""

    analyzer = SentimentAnalyzer(model=model, client=client)
    try:
        return analyzer.analyze(text).model_dump()
    finally:
        analyzer.close()
