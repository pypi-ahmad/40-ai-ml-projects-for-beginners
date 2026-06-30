"""Optional integration tests that require running Ollama daemon."""

from __future__ import annotations

import pytest

from streamlit_app.utils.models import analyze_sentiment, is_ollama_available


@pytest.mark.skipif(not is_ollama_available(), reason="Ollama daemon unavailable")
def test_live_sentiment_inference() -> None:
    result = analyze_sentiment("The user interface is clean and responsive.")
    assert result["sentiment"] in {"positive", "neutral", "negative"}
    assert 0.0 <= float(result["confidence"]) <= 1.0
