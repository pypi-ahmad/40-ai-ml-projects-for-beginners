"""Helper utilities for the AI application."""

import json
import re
import hashlib
from typing import Optional

from streamlit_app.utils.config import get_settings


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length with ellipsis.

    Long text needs to be truncated for display in:
      - DataFrames and tables
      - Summary cards
      - UI components with limited space

    Args:
        text: Input text
        max_length: Maximum characters before truncation

    Returns:
        Truncated text with "..." if needed
    """
    if max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def compute_text_hash(text: str) -> str:
    """
    Compute a stable hash of text for caching and deduplication.

    Hash-based keys are more reliable than raw text for:
      - Cache keys (avoiding long string keys)
      - Deduplication (same text = same hash)
      - Rate limiting (tracking unique requests)

    Args:
        text: Input text

    Returns:
        SHA-256 hex digest (first 16 characters)
    """
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def estimate_reading_time(text: str) -> int:
    """
    Estimate reading time in whole minutes (200 words per minute).

    Useful for:
      - Showing processing time estimates
      - Setting appropriate timeout values
      - User experience feedback

    Args:
        text: Input text

    Returns:
        Estimated reading time in minutes
    """
    word_count = len(text.split())
    if word_count == 0:
        return 0
    return max(1, round(word_count / 200))


def validate_text_input(
    text: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
) -> Optional[str]:
    """
    Validate user text input before processing.

    Input validation is a critical security practice:
      - Prevent processing empty or garbage input
      - Enforce reasonable length limits
      - Prevent injection attacks
      - Provide clear error messages to users

    Args:
        text: Input text to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length

    Returns:
        Error message string if invalid, None if valid
    """
    limits = get_settings().limits
    min_len = limits.min_text_length if min_length is None else min_length
    max_len = limits.max_text_length if max_length is None else max_length

    if not text or not text.strip():
        return "Please enter some text."

    if len(text.strip()) < min_len:
        return f"Text is too short. Minimum {min_len} characters required."

    if len(text) > max_len:
        return f"Text is too long. Maximum {max_len} characters allowed."

    return None


def validate_categories(categories: list[str]) -> Optional[str]:
    """Validate user-provided classification categories."""
    limits = get_settings().limits

    if len(categories) < 2:
        return "Please provide at least 2 categories."

    if len(categories) > limits.max_categories:
        return f"Please provide at most {limits.max_categories} categories."

    for category in categories:
        if len(category) > limits.max_category_length:
            return (
                f"Category '{category[:24]}...' is too long. "
                f"Max {limits.max_category_length} characters per category."
            )

    return None


def format_confidence(score: float) -> str:
    """
    Format a confidence score as a percentage string.

    Args:
        score: Confidence score between 0 and 1

    Returns:
        Formatted percentage string (e.g. "87.3%")
    """
    return f"{round(score * 100)}%"


def get_status_emoji(label: str) -> str:
    """
    Return an emoji for a sentiment or status label.

    Visual indicators make results more scannable:
      - POSITIVE: green/good
      - NEGATIVE: red/bad
      - NEUTRAL: yellow/neutral

    Args:
        label: Status label (POSITIVE, NEGATIVE, NEUTRAL, etc.)

    Returns:
        Emoji string
    """
    mapping = {
        "POSITIVE": "🟢",
        "NEGATIVE": "🔴",
        "NEUTRAL": "🟡",
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
        "info": "ℹ️",
    }
    return mapping.get(label.upper(), "ℹ️")


def safe_json_parse(data: str) -> Optional[dict]:
    """
    Safely parse a JSON string, returning None on failure.

    Args:
        data: JSON string to parse

    Returns:
        Parsed dict or None
    """
    if not data:
        return None

    # Direct parse first for clean JSON payloads.
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        pass

    # Strip fenced code blocks like ```json ... ```.
    fenced_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", data, re.IGNORECASE)
    if fenced_match:
        try:
            return json.loads(fenced_match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass

    # Try first JSON object embedded in surrounding text.
    object_match = re.search(r"\{[\s\S]*\}", data)
    if object_match:
        try:
            return json.loads(object_match.group(0))
        except (json.JSONDecodeError, TypeError):
            return None

    return None


def validate_input(text: str, min_length: int = 3, max_length: int = 5000) -> Optional[str]:
    """Backward-compatible alias used by older tests/material."""
    return validate_text_input(text, min_length=min_length, max_length=max_length)


def map_sentiment_to_emoji(label: str) -> str:
    """Backward-compatible alias used by older tests/material."""
    return get_status_emoji(label)


def parse_llm_json_response(data: str) -> Optional[dict]:
    """Backward-compatible alias used by older tests/material."""
    return safe_json_parse(data)
