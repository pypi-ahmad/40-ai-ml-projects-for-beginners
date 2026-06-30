"""Prompt and response safety checks."""

from __future__ import annotations

from llmft.config.schemas import SafetyConfig

TOXIC_TERMS = {"kill", "hate", "terror"}


def validate_prompt(prompt: str, config: SafetyConfig) -> tuple[bool, str]:
    """Validate prompt against configured safety patterns."""
    lowered = prompt.lower()
    for pattern in config.banned_patterns:
        if pattern.lower() in lowered:
            return False, f"blocked pattern: {pattern}"
    if len(prompt.strip()) == 0:
        return False, "empty prompt"
    return True, "ok"


def sanitize_dataset_rows(rows: list[dict[str, str]], config: SafetyConfig) -> list[dict[str, str]]:
    """Filter out rows containing banned patterns."""
    cleaned: list[dict[str, str]] = []
    for row in rows:
        blob = " ".join(row.values()).lower()
        if any(pattern.lower() in blob for pattern in config.banned_patterns):
            continue
        cleaned.append(row)
    return cleaned


def detect_unsafe_response(text: str, config: SafetyConfig) -> tuple[bool, str]:
    """Basic toxicity heuristic."""
    if not config.enable_toxicity_check:
        return False, "toxicity check disabled"
    lowered = text.lower()
    for term in TOXIC_TERMS:
        if term in lowered:
            return True, f"toxic term detected: {term}"
    return False, "safe"
