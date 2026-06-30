"""Tests for helper utilities (non-UI logic)."""

from streamlit_app.utils.helpers import (
    format_latency,
    load_sample_text,
    now_utc_iso,
    truncate_text,
    validate_categories,
    validate_text_input,
)


def test_truncate_text_short_enough() -> None:
    assert truncate_text("hello", 10) == "hello"


def test_truncate_text_long() -> None:
    result = truncate_text("this is a very long sentence to truncate", 15)
    assert result.endswith("...")


def test_validate_text_input_none() -> None:
    assert validate_text_input(None) == "Please enter some text."


def test_validate_text_input_too_short() -> None:
    assert validate_text_input("ab", 3) == "Text must be at least 3 characters."


def test_validate_text_input_valid() -> None:
    assert validate_text_input("hello world") is None


def test_validate_categories_requires_two() -> None:
    assert validate_categories(["OnlyOne"]) == "Provide at least 2 categories for classification."


def test_validate_categories_unique() -> None:
    assert validate_categories(["A", "A"]) == "Categories must be unique."


def test_validate_categories_valid() -> None:
    assert validate_categories(["A", "B", "C"]) is None


def test_format_latency_units() -> None:
    assert "ms" in format_latency(0.25)
    assert "s" in format_latency(2.1)
    assert "min" in format_latency(121)


def test_now_utc_iso() -> None:
    value = now_utc_iso()
    assert "T" in value


def test_load_sample_text_valid_key() -> None:
    text = load_sample_text("summary")
    assert isinstance(text, str)
    assert len(text) > 20


def test_load_sample_text_invalid_key() -> None:
    assert load_sample_text("nonexistent") == "Sample text not found."
