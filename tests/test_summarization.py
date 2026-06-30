from unittest.mock import MagicMock, patch

import pytest

from src.summarization import Summarizer


@pytest.fixture
def summarizer() -> Summarizer:
    s = Summarizer()
    yield s
    s.close()


def test_parse_valid_json() -> None:
    raw = '{"summary": "short", "key_points": ["a", "b"], "tldr": "done"}'
    result = Summarizer._parse(raw)
    assert result["summary"] == "short"
    assert result["key_points"] == ["a", "b"]
    assert result["tldr"] == "done"


def test_parse_json_with_markdown_fence() -> None:
    raw = '```json\n{"summary": "ok", "key_points": [], "tldr": "x"}\n```'
    result = Summarizer._parse(raw)
    assert result["summary"] == "ok"


def test_parse_fallback_on_garbage() -> None:
    result = Summarizer._parse("not json")
    assert result["summary"] == "not json"
    assert result["key_points"] == []
    assert result["tldr"] == "parse failed"


def test_parse_empty_string() -> None:
    result = Summarizer._parse("")
    assert result["tldr"] == "parse failed"


@patch("src.summarization.OllamaClient")
def test_summarize_short_text_no_api_call(mock_client: MagicMock) -> None:
    s = Summarizer()
    result = s.summarize("Short text")
    assert result["summary"] == "Short text"
    mock_client.return_value.generate.assert_not_called()
    s.close()


@patch("src.summarization.OllamaClient")
def test_summarize_makes_api_call(mock_ollama: MagicMock) -> None:
    mock_ollama.return_value.generate.return_value = {
        "response": '{"summary": "long doc", "key_points": ["p1"], "tldr": "done"}'
    }
    s = Summarizer()
    text = "A" * 100
    result = s.summarize(text)
    assert result["summary"] == "long doc"
    mock_ollama.return_value.generate.assert_called_once()
    s.close()


@patch("src.summarization.OllamaClient")
def test_summarize_api_fallback_on_bad_response(mock_ollama: MagicMock) -> None:
    mock_ollama.return_value.generate.return_value = {"response": "nope"}
    s = Summarizer()
    result = s.summarize("A" * 60)
    assert result["tldr"] == "parse failed"
    s.close()


def test_summarize_empty_text() -> None:
    s = Summarizer()
    result = s.summarize("")
    assert result["summary"] == ""
    s.close()


def test_summarize_whitespace_only() -> None:
    s = Summarizer()
    result = s.summarize("   ")
    assert result["summary"] == "   "
    s.close()
