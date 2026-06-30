"""Integration tests hitting local Ollama runtime with real model inference."""

from __future__ import annotations

import httpx
import pytest

from src.benchmarking import BenchmarkRunner
from src.config import get_config
from src.ollama_client import OllamaClient
from src.sentiment import SentimentAnalyzer
from src.translation import Translator

REQUIRED_MODELS = {
    "qwen3.5:2b",
    "qwen3.5:4b",
    "granite4.1:3b",
    "nemotron-3-nano:4b",
    "translategemma:4b",
    "glm-ocr:latest",
    "deepseek-ocr:latest",
}


def _discover_installed_models() -> set[str]:
    """Return installed Ollama models; empty set means runtime unavailable."""

    client = OllamaClient()
    try:
        return set(client.list_models())
    finally:
        client.close()


def _ollama_reachable() -> bool:
    """Probe Ollama tags endpoint to distinguish unreachable vs no-model states."""

    base_url = get_config().ollama_base_url.rstrip("/")
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        return response.status_code == 200
    except httpx.RequestError:
        return False


OLLAMA_REACHABLE = _ollama_reachable()
INSTALLED_MODELS = _discover_installed_models() if OLLAMA_REACHABLE else set()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not OLLAMA_REACHABLE,
        reason="Ollama is not reachable from test environment; skipping live integration tests.",
    ),
]


def test_required_models_available_locally() -> None:
    missing = REQUIRED_MODELS - INSTALLED_MODELS
    assert not missing, f"Missing models: {sorted(missing)}"


def test_live_sentiment_inference() -> None:
    analyzer = SentimentAnalyzer()
    try:
        result = analyzer.analyze("I love how smooth and reliable this workflow feels.")
    finally:
        analyzer.close()

    assert result.error is None
    assert result.label in {"Positive", "Neutral", "Negative"}
    assert result.confidence >= 0.0


def test_live_translation_inference() -> None:
    translator = Translator()
    try:
        result = translator.translate("Good morning. How are you today?", "English", "Spanish")
    finally:
        translator.close()

    assert result.error is None
    assert len(result.translated_text) > 0


def test_live_single_benchmark_run() -> None:
    runner = BenchmarkRunner()
    try:
        result = runner.run_single(model="qwen3.5:2b", prompt_key="short", runs=1)
    finally:
        runner.close()

    assert result.successful_runs >= 1
    assert result.mean_latency_ms > 0
