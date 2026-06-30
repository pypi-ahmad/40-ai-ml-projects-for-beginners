"""Explainability utilities for generation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TokenTrace:
    """Token-level confidence approximation."""

    token: str
    confidence: float


def token_confidence_trace(text: str) -> list[TokenTrace]:
    """Create pseudo confidence trace for generated text.

    Args:
        text: Generated text.

    Returns:
        List of token confidence entries.

    Example:
        >>> traces = token_confidence_trace("hello world")
        >>> len(traces)
        2
    """
    tokens = text.split()
    out: list[TokenTrace] = []
    for idx, token in enumerate(tokens):
        confidence = max(0.05, 0.95 - idx * 0.03)
        out.append(TokenTrace(token=token, confidence=round(confidence, 3)))
    return out
