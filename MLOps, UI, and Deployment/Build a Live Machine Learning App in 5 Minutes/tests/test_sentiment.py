"""Unit tests for sentiment module parsing and validation behavior."""

from __future__ import annotations

from src.sentiment import SentimentAnalyzer


class StubClient:
    """Deterministic stub for Ollama client responses."""

    def __init__(self, response: str = "") -> None:
        self._response = response

    def generate(self, **_: object) -> dict[str, object]:
        return {
            "response": self._response,
            "latency_ms": 12.0,
            "eval_count": 40,
            "eval_duration_ns": 100,
            "error": None,
        }

    def close(self) -> None:
        pass


def test_sentiment_empty_input_returns_validation_error() -> None:
    analyzer = SentimentAnalyzer(client=StubClient())
    result = analyzer.analyze("   ")
    assert result.error is not None
    assert result.error.stage == "validation"


def test_sentiment_parses_json_response() -> None:
    client = StubClient(
        '{"label":"Positive","confidence":0.91,"explanation":"Strong positive wording."}'
    )
    analyzer = SentimentAnalyzer(client=client)
    result = analyzer.analyze("I love this product!")

    assert result.error is None
    assert result.label == "Positive"
    assert result.confidence == 0.91
    assert "positive" in result.explanation.lower()


def test_sentiment_falls_back_to_heuristic() -> None:
    client = StubClient("This is excellent and great.")
    analyzer = SentimentAnalyzer(client=client)
    result = analyzer.analyze("Amazing quality")

    assert result.error is None
    assert result.label == "Positive"
