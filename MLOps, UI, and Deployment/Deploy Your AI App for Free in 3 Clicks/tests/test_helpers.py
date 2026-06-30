"""
Tests for helpers.py — utility functions.

Covers:
  - Text truncation, hashing, reading time, validation
  - Confidence formatting, emoji mapping, JSON parsing
  - Edge cases and error handling
"""

import pytest
from streamlit_app.utils.helpers import (
    truncate_text,
    compute_text_hash,
    estimate_reading_time,
    validate_text_input,
    validate_categories,
    format_confidence,
    get_status_emoji,
    safe_json_parse,
)


class TestTruncateText:
    def test_truncates_long_text(self):
        text = "a" * 1000
        result = truncate_text(text, 100)
        assert len(result) == 100

    def test_keeps_short_text(self):
        text = "Short text"
        result = truncate_text(text, 100)
        assert result == text

    def test_exact_length(self):
        text = "a" * 50
        result = truncate_text(text, 50)
        assert result == text

    def test_zero_max_length(self):
        text = "Some text"
        result = truncate_text(text, 0)
        assert result == ""

    def test_negative_max_length(self):
        text = "Some text"
        result = truncate_text(text, -1)
        assert result == ""

    def test_empty_string(self):
        assert truncate_text("", 100) == ""


class TestComputeTextHash:
    def test_returns_string(self):
        result = compute_text_hash("hello")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_deterministic(self):
        assert compute_text_hash("test") == compute_text_hash("test")

    def test_different_inputs_different_hashes(self):
        assert compute_text_hash("abc") != compute_text_hash("xyz")

    def test_empty_string_has_hash(self):
        result = compute_text_hash("")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_long_text(self):
        text = "x" * 10000
        result = compute_text_hash(text)
        assert isinstance(result, str)
        assert len(result) > 0


class TestEstimateReadingTime:
    def test_returns_int(self):
        result = estimate_reading_time("Hello world")
        assert isinstance(result, int)

    def test_zero_for_empty(self):
        assert estimate_reading_time("") == 0

    def test_positive_for_long_text(self, sample_texts):
        result = estimate_reading_time(sample_texts["long_article"])
        assert result > 0

    def test_proportional_to_length(self):
        short_result = estimate_reading_time("word " * 50)
        long_result = estimate_reading_time("word " * 500)
        assert long_result > short_result

    def test_reasonable_range(self):
        result = estimate_reading_time("word " * 200)
        assert 0 < result < 30


class TestValidateTextInput:
    def test_valid_text(self):
        result = validate_text_input("Hello world", min_length=1, max_length=1000)
        assert result is None

    def test_too_short(self):
        result = validate_text_input("Hi", min_length=10, max_length=1000)
        assert result is not None
        assert "short" in result.lower()

    def test_too_long(self):
        text = "a" * 2000
        result = validate_text_input(text, min_length=1, max_length=100)
        assert result is not None
        assert "long" in result.lower()

    def test_empty_text(self):
        result = validate_text_input("", min_length=1, max_length=1000)
        assert result is not None

    def test_exact_min_length(self):
        text = "a" * 10
        result = validate_text_input(text, min_length=10, max_length=100)
        assert result is None

    def test_exact_max_length(self):
        text = "a" * 100
        result = validate_text_input(text, min_length=1, max_length=100)
        assert result is None

    def test_default_parameters(self):
        result = validate_text_input("Hello world")
        assert result is None


class TestValidateCategories:
    def test_valid_categories(self):
        assert validate_categories(["technology", "health"]) is None

    def test_rejects_single_category(self):
        error = validate_categories(["technology"])
        assert error is not None
        assert "at least 2" in error.lower()

    def test_rejects_too_many_categories(self):
        categories = [f"cat-{i}" for i in range(30)]
        error = validate_categories(categories)
        assert error is not None
        assert "at most" in error.lower()

    def test_rejects_very_long_label(self):
        error = validate_categories(["technology", "x" * 80])
        assert error is not None
        assert "too long" in error.lower()


class TestFormatConfidence:
    def test_returns_string_with_percent(self):
        result = format_confidence(0.85)
        assert isinstance(result, str)
        assert "%" in result

    def test_zero(self):
        assert "0%" in format_confidence(0.0)

    def test_one(self):
        assert "100%" in format_confidence(1.0)

    def test_mid_range(self):
        result = format_confidence(0.5)
        assert "50" in result

    def test_edge_high(self):
        assert "100" in format_confidence(0.9999)

    def test_edge_low(self):
        assert "0" in format_confidence(0.0001)


class TestGetStatusEmoji:
    def test_positive_returns_emoji(self):
        result = get_status_emoji("positive")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_negative_returns_emoji(self):
        result = get_status_emoji("negative")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_neutral_returns_emoji(self):
        result = get_status_emoji("neutral")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_case_insensitive(self):
        assert get_status_emoji("POSITIVE") == get_status_emoji("positive")

    def test_unknown_returns_default(self):
        result = get_status_emoji("unknown")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_string_returns_default(self):
        result = get_status_emoji("")
        assert isinstance(result, str)


class TestSafeJsonParse:
    def test_parses_valid_json(self):
        result = safe_json_parse('{"key": "value"}')
        assert isinstance(result, dict)
        assert result["key"] == "value"

    def test_handles_json_in_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = safe_json_parse(text)
        assert result["key"] == "value"

    def test_handles_json_with_surrounding_text(self):
        text = 'Here is the result:\n{"answer": 42}\nHope that helps.'
        result = safe_json_parse(text)
        assert result["answer"] == 42

    def test_returns_none_for_non_json(self):
        result = safe_json_parse("Just some random text")
        assert result is None

    def test_returns_none_for_empty(self):
        assert safe_json_parse("") is None

    def test_parses_nested_json(self):
        json_str = '{"outer": {"inner": "value"}, "list": [1, 2, 3]}'
        result = safe_json_parse(json_str)
        assert result["outer"]["inner"] == "value"
        assert result["list"] == [1, 2, 3]
