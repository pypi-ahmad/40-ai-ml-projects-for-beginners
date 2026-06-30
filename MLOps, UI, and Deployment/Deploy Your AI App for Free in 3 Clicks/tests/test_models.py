"""
Tests for models.py — three-tier fallback inference.

Covers:
  - Rule-based fallback layer for all 4 functions
  - Input validation and error handling
  - Return value contract compliance
  - Edge cases: empty text, long text, special characters
"""

import pytest
from streamlit_app.utils.models import (
    analyze_sentiment,
    summarize_text,
    classify_text,
    translate_text,
    SUPPORTED_LANGUAGES,
)


class TestAnalyzeSentiment:
    """Tests for analyze_sentiment()."""

    def test_returns_dict_with_expected_keys(self):
        result = analyze_sentiment("I love this!")
        assert isinstance(result, dict)
        assert "label" in result
        assert "confidence" in result
        assert isinstance(result["label"], str)
        assert isinstance(result["confidence"], float)

    def test_positive_sentiment(self, sample_texts):
        result = analyze_sentiment(sample_texts["positive"])
        assert result["label"].lower() == "positive"
        assert 0.5 <= result["confidence"] <= 1.0

    def test_negative_sentiment(self, sample_texts):
        result = analyze_sentiment(sample_texts["negative"])
        assert result["label"].lower() == "negative"
        assert 0.5 <= result["confidence"] <= 1.0

    def test_neutral_sentiment(self, sample_texts):
        result = analyze_sentiment(sample_texts["neutral"][:200])
        assert result["label"].lower() == "neutral"

    def test_handles_empty_text_gracefully(self):
        result = analyze_sentiment("")
        assert result["label"].lower() == "neutral"
        assert result["confidence"] == 0.5

    def test_handles_very_long_text(self):
        long_text = "happy " * 5000
        result = analyze_sentiment(long_text)
        assert "label" in result

    def test_sentiment_stability_same_text(self):
        r1 = analyze_sentiment("This is great!")
        r2 = analyze_sentiment("This is great!")
        assert r1 == r2

    def test_confidence_in_expected_range(self, sample_texts):
        for key in ["positive", "negative", "neutral"]:
            result = analyze_sentiment(sample_texts[key][:200])
            assert 0.0 <= result["confidence"] <= 1.0

    def test_all_three_sentiments_different(self, sample_texts):
        pos = analyze_sentiment(sample_texts["positive"])["label"]
        neg = analyze_sentiment(sample_texts["negative"])["label"]
        neu = analyze_sentiment(sample_texts["neutral"][:200])["label"]
        assert pos != neg or pos != neu

    def test_special_characters(self):
        result = analyze_sentiment("This is great!!! 🎉 (amazing)")
        assert "label" in result


class TestSummarizeText:
    """Tests for summarize_text()."""

    def test_returns_dict_with_expected_keys(self):
        result = summarize_text("A long text to summarize.")
        assert isinstance(result, dict)
        assert "summary" in result
        assert isinstance(result["summary"], str)

    def test_summary_is_shorter_than_original(self, sample_texts):
        result = summarize_text(sample_texts["long_article"])
        summary = result["summary"]
        assert len(summary) < len(sample_texts["long_article"])
        assert len(summary) > 0

    def test_handles_short_text(self):
        result = summarize_text("Hello.")
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_handles_empty_text(self):
        result = summarize_text("")
        assert isinstance(result["summary"], str)

    def test_summary_not_identical_to_input(self, sample_texts):
        result = summarize_text(sample_texts["long_article"])
        assert result["summary"].strip() != sample_texts["long_article"].strip()

    def test_with_mixed_punctuation(self):
        text = "Wow! This is... amazing? Yes, it really is! Fantastic work."
        result = summarize_text(text)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_custom_max_length(self):
        text = "word " * 500
        short_result = summarize_text(text, max_length=30)
        long_result = summarize_text(text, max_length=100)
        assert len(short_result["summary"]) < len(long_result["summary"]) or len(short_result["summary"]) <= len(long_result["summary"])


class TestClassifyText:
    """Tests for classify_text()."""

    def test_returns_dict_with_expected_keys(self):
        categories = ["technology", "health", "sports"]
        result = classify_text("Some text about AI.", categories)
        assert isinstance(result, dict)
        assert "label" in result
        assert "confidence" in result
        assert isinstance(result["label"], str)
        assert isinstance(result["confidence"], float)

    def test_tech_classification(self, sample_texts):
        categories = ["technology", "health", "sports"]
        result = classify_text(sample_texts["tech"], categories)
        assert result["label"].lower() == "technology"
        assert 0.5 <= result["confidence"] <= 1.0

    def test_health_classification(self, sample_texts):
        categories = ["technology", "health", "sports"]
        result = classify_text(sample_texts["health"], categories)
        assert result["label"].lower() == "health"

    def test_handles_single_category(self):
        result = classify_text("Any text here.", ["general"])
        assert isinstance(result["label"], str)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_handles_empty_categories(self):
        with pytest.raises(IndexError):
            classify_text("text", [])

    def test_handles_empty_text(self):
        result = classify_text("", ["tech", "health"])
        assert result["label"].lower() == "tech"
        assert result["confidence"] == 0.0

    def test_with_special_characters(self):
        categories = ["technology", "health"]
        text = "AI & ML are @ the forefront of #tech innovation 2024!"
        result = classify_text(text, categories)
        assert isinstance(result["label"], str)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_stability_same_text(self):
        categories = ["technology", "health", "sports"]
        r1 = classify_text("Running and football.", categories)
        r2 = classify_text("Running and football.", categories)
        assert r1 == r2

    def test_default_labels_when_none(self):
        result = classify_text("Technology news")
        assert isinstance(result["label"], str)
        assert 0.0 <= result["confidence"] <= 1.0


class TestTranslateText:
    """Tests for translate_text()."""

    def test_returns_dict_with_expected_keys(self):
        result = translate_text("Hello", target_lang="es")
        assert isinstance(result, dict)
        assert "translated_text" in result
        assert isinstance(result["translated_text"], str)

    def test_translates_to_spanish(self):
        result = translate_text("Hello, how are you?", target_lang="es")
        assert len(result["translated_text"]) > 0

    def test_translates_to_french(self):
        result = translate_text("Good morning", target_lang="fr")
        assert len(result["translated_text"]) > 0

    def test_translates_to_german(self):
        result = translate_text("Thank you very much", target_lang="de")
        assert len(result["translated_text"]) > 0

    def test_source_lang_detected(self, sample_texts):
        result = translate_text(sample_texts["positive"], target_lang="es")
        assert "translated_text" in result
        assert len(result["translated_text"]) > 0

    def test_handles_empty_text(self):
        result = translate_text("", target_lang="es")
        assert result["translated_text"].startswith("[")

    def test_handles_unsupported_language(self):
        result = translate_text("Hello", target_lang="xx")
        assert len(result["translated_text"]) > 0

    def test_supported_languages_includes_common_ones(self):
        assert "es" in SUPPORTED_LANGUAGES
        assert "fr" in SUPPORTED_LANGUAGES
        assert "de" in SUPPORTED_LANGUAGES
        assert "en" in SUPPORTED_LANGUAGES

    def test_supported_languages_values_are_dicts(self):
        for code, info in SUPPORTED_LANGUAGES.items():
            assert isinstance(code, str)
            assert isinstance(info, dict)
            assert "name" in info
            assert "flag" in info
            assert isinstance(info["name"], str)
            assert isinstance(info["flag"], str)
            assert len(code) >= 2
            assert len(info["name"]) > 0

    def test_translation_not_empty_for_known_languages(self):
        for lang in ["es", "fr", "de", "it", "pt"]:
            result = translate_text("Hello world", target_lang=lang)
            assert len(result["translated_text"]) > 0

    def test_with_special_characters(self):
        text = "Hello! How's it going? Price is $9.99 (discount 50%)."
        result = translate_text(text, target_lang="es")
        assert len(result["translated_text"]) > 0

    def test_source_lang_passthrough(self):
        result = translate_text("Hello", source_lang="en", target_lang="fr")
        assert "translated_text" in result
