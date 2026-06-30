"""Tests for caching wrappers and cache comparison helper."""

from __future__ import annotations

from unittest.mock import patch

from streamlit_app.utils.caching import cached_sentiment, compare_cached_vs_uncached_sentiment


@patch("streamlit_app.utils.caching.models.analyze_sentiment")
def test_cached_sentiment_calls_model(mock_analyze) -> None:
    mock_analyze.return_value = {"sentiment": "positive", "confidence": 0.9, "explanation": "ok"}
    cached_sentiment.clear()
    result = cached_sentiment("great product", "granite4.1:3b")
    assert result["sentiment"] == "positive"
    mock_analyze.assert_called_once()


@patch("streamlit_app.utils.caching.models.analyze_sentiment")
def test_compare_cached_vs_uncached_returns_timings(mock_analyze) -> None:
    mock_analyze.return_value = {"sentiment": "neutral", "confidence": 0.5, "explanation": "ok"}
    cached_sentiment.clear()
    metrics = compare_cached_vs_uncached_sentiment("text", "granite4.1:3b")
    assert set(metrics.keys()) == {"uncached_seconds", "cached_first_seconds", "cached_second_seconds"}
    for value in metrics.values():
        assert value >= 0
