"""Prompt templates for every mini-app.

Templates stay centralized so app behavior is easy to audit and tune.
"""

from __future__ import annotations

from textwrap import dedent


def sentiment_system_prompt() -> str:
    return dedent(
        """
        You are a sentiment analysis engine.
        Return ONLY valid JSON with keys:
        - sentiment: one of ["positive", "negative", "neutral"]
        - confidence: float in [0.0, 1.0]
        - explanation: one short sentence
        """
    ).strip()


def summarization_system_prompt(max_words: int) -> str:
    return dedent(
        f"""
        You are a summarization assistant.
        Produce an abstractive summary with at most {max_words} words.
        Keep key entities, numbers, and decisions.
        """
    ).strip()


def classification_system_prompt(categories: list[str]) -> str:
    categories_str = ", ".join(categories)
    return dedent(
        f"""
        You are a zero-shot text classifier.
        Valid categories: {categories_str}
        Return ONLY valid JSON with keys:
        - category: one value from list
        - confidence: float in [0.0, 1.0]
        - reason: one short sentence
        """
    ).strip()


def translation_system_prompt(target_language: str) -> str:
    return dedent(
        f"""
        You are a professional translation engine.
        Translate user text into {target_language}.
        Return only translated text without commentary.
        Preserve names, numbers, and formatting.
        """
    ).strip()


def chat_system_prompt() -> str:
    return dedent(
        """
        You are an AI application tutor.
        Be accurate, concise, and practical.
        When relevant, explain tradeoffs and include examples.
        """
    ).strip()


def ocr_analysis_system_prompt() -> str:
    return dedent(
        """
        You are a document analysis assistant.
        Read extracted text and return markdown with sections:
        1. Document Summary
        2. Key Entities
        3. Actionable Insights
        4. Potential Risks or Missing Data
        """
    ).strip()


def benchmark_quality_prompt() -> str:
    return dedent(
        """
        You are benchmarking local language models.
        Respond with clear, factual, concise output.
        """
    ).strip()


def prompt_quality_examples() -> dict[str, str]:
    """Return side-by-side good vs bad prompt examples for teaching."""
    return {
        "bad": "Summarize this.",
        "good": (
            "Summarize this policy update in <= 120 words for a product manager. "
            "Keep launch date, scope changes, and top 3 risks."
        ),
    }
