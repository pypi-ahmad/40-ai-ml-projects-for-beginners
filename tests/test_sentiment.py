from unittest.mock import MagicMock, patch

import pytest

from src.sentiment import SentimentAnalyzer


@pytest.fixture
def analyzer() -> SentimentAnalyzer:
    a = SentimentAnalyzer()
    yield a
    a.close()


def test_parse_valid_json() -> None:
    raw = '{"label": "positive", "score": 0.95, "explanation": "clear positive"}'
    result = SentimentAnalyzer._parse(raw)
    assert result["label"] == "positive"
    assert result["score"] == 0.95


def test_parse_json_with_markdown_fence() -> None:
    raw = '```json\n{"label": "negative", "score": 0.88, "explanation": "negative words"}\n```'
    result = SentimentAnalyzer._parse(raw)
    assert result["label"] == "negative"


def test_parse_fallback_on_garbage() -> None:
    result = SentimentAnalyzer._parse("not json at all")
    assert result["label"] == "neutral"
    assert result["score"] == 0.0


@patch("src.sentiment.OllamaClient")
def test_analyzer_returns_expected_keys(mock_ollama: MagicMock) -> None:
    mock_ollama.return_value.generate.return_value = {
        "response": '{"label": "positive", "score": 0.95, "explanation": "mock"}'
    }
    a = SentimentAnalyzer()
    result = a.analyze("I love this!")
    assert "label" in result
    assert "score" in result
    assert "explanation" in result
    assert result["label"] in ("positive", "negative", "neutral")
    a.close()


@patch("src.sentiment.OllamaClient")
def test_analyze_batch(mock_ollama: MagicMock) -> None:
    mock_ollama.return_value.generate.return_value = {
        "response": '{"label": "neutral", "score": 0.5, "explanation": "mock"}'
    }
    a = SentimentAnalyzer()
    results = a.analyze_batch(["good", "bad", "okay"])
    assert len(results) == 3
    for r in results:
        assert "label" in r
    a.close()


@patch("src.sentiment.OllamaClient")
def test_empty_text(mock_ollama: MagicMock) -> None:
    mock_ollama.return_value.generate.return_value = {
        "response": '{"label": "neutral", "score": 0.5, "explanation": "empty"}'
    }
    a = SentimentAnalyzer()
    result = a.analyze("")
    assert isinstance(result["label"], str)
    a.close()
