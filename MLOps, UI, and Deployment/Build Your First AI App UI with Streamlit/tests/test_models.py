"""Tests for model integration functions with mocked Ollama API."""

from __future__ import annotations

import json
from unittest.mock import patch

from streamlit_app.utils.models import (
    _call_ollama,
    _try_parse_json,
    analyze_sentiment,
    benchmark_inference,
    chat_response,
    classify_text,
    get_embedding,
    summarize_text,
    translate_text,
)


@patch("streamlit_app.utils.models.ollama.chat")
def test_call_ollama_success(mock_chat) -> None:
    mock_chat.return_value = {"message": {"content": "  hello world  "}}
    result = _call_ollama("qwen3.5:4b", "prompt", "system")
    assert result == "hello world"


@patch("streamlit_app.utils.models.ollama.chat")
def test_call_ollama_error(mock_chat) -> None:
    mock_chat.side_effect = ConnectionError("Ollama not running")
    result = _call_ollama("qwen3.5:4b", "prompt")
    assert result.startswith("Error:")


def test_try_parse_json_success() -> None:
    raw = '{"sentiment": "positive", "confidence": 0.9}'
    parsed = _try_parse_json(raw, {"fallback": True})
    assert parsed["sentiment"] == "positive"


def test_try_parse_json_fallback() -> None:
    parsed = _try_parse_json("no json", {"fallback": True})
    assert parsed["fallback"] is True


@patch("streamlit_app.utils.models._call_ollama")
def test_analyze_sentiment_valid_json(mock_call) -> None:
    mock_call.return_value = json.dumps(
        {"sentiment": "negative", "confidence": 0.8, "explanation": "Contains negative words."}
    )
    result = analyze_sentiment("bad experience")
    assert result["sentiment"] == "negative"
    assert result["confidence"] == 0.8


@patch("streamlit_app.utils.models._call_ollama")
def test_analyze_sentiment_parse_failure(mock_call) -> None:
    mock_call.return_value = "raw text only"
    result = analyze_sentiment("test")
    assert result["sentiment"] == "neutral"


@patch("streamlit_app.utils.models._call_ollama")
def test_summarize_text(mock_call) -> None:
    mock_call.return_value = "short summary"
    result = summarize_text("long text" * 10, 100)
    assert result == "short summary"


@patch("streamlit_app.utils.models._call_ollama")
def test_classify_text(mock_call) -> None:
    mock_call.return_value = json.dumps(
        {"category": "Technology", "confidence": 0.7, "reason": "Topic is AI tooling."}
    )
    result = classify_text("AI app architecture", ["Technology", "Finance"])
    assert result["category"] == "Technology"


@patch("streamlit_app.utils.models._call_ollama")
def test_translate_text(mock_call) -> None:
    mock_call.return_value = "Bonjour"
    result = translate_text("Hello", "French")
    assert result == "Bonjour"


@patch("streamlit_app.utils.models.ollama.chat")
def test_chat_response_success(mock_chat) -> None:
    mock_chat.return_value = {"message": {"content": "assistant reply"}}
    result = chat_response([{"role": "user", "content": "hi"}])
    assert result == "assistant reply"


@patch("streamlit_app.utils.models.ollama.chat")
def test_chat_response_error(mock_chat) -> None:
    mock_chat.side_effect = RuntimeError("fail")
    result = chat_response([{"role": "user", "content": "hi"}])
    assert result.startswith("Error:")


@patch("streamlit_app.utils.models._call_ollama")
def test_benchmark_inference_fields(mock_call) -> None:
    mock_call.return_value = "benchmark output"
    result = benchmark_inference("qwen3.5:4b", "prompt", runs=2)
    assert result["model"] == "qwen3.5:4b"
    assert "mean_latency" in result
    assert "mean_throughput_wps" in result
    assert len(result["outputs"]) == 2
    assert len(result["records"]) == 2


@patch("streamlit_app.utils.models.ollama.embed", create=True)
def test_get_embedding_embed_api(mock_embed) -> None:
    mock_embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
    result = get_embedding("hello")
    assert result == [0.1, 0.2, 0.3]


@patch("streamlit_app.utils.models.ollama.embed", create=True)
def test_get_embedding_error(mock_embed) -> None:
    mock_embed.side_effect = RuntimeError("fail")
    result = get_embedding("hello")
    assert result == []
