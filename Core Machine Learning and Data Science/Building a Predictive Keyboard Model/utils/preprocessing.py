"""Backward-compatible preprocessing helpers."""

from __future__ import annotations

from pathlib import Path

from .tokenization import RegexTokenizerBackend, normalize_text


def clean_text(text: str, lower: bool = True) -> str:
    """Normalize text using shared normalization pipeline."""

    return normalize_text(text, lowercase=lower)


def tokenize(text: str) -> list[str]:
    """Regex tokenization helper retained for compatibility."""

    return RegexTokenizerBackend().tokenize(text)


def load_corpus(filepath: str, lower: bool = True) -> str:
    """Load text corpus from path."""

    return clean_text(Path(filepath).read_text(encoding="utf-8", errors="replace"), lower=lower)


load_text = load_corpus


def sentence_tokenize(text: str) -> list[str]:
    """Split text into sentences using punctuation boundary heuristic."""

    import re

    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]
