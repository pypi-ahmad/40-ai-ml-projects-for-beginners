"""Pluggable explainability methods (SHAP, LIME, integrated gradients)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def integrated_gradients(
    embedding_fn: Callable[[str], np.ndarray],
    baseline_text: str,
    text: str,
    steps: int = 16,
) -> np.ndarray:
    """Approximate integrated gradients in embedding space."""
    baseline = embedding_fn(baseline_text)
    target = embedding_fn(text)
    grads = []
    for alpha in np.linspace(0.0, 1.0, steps):
        interpolated = baseline + alpha * (target - baseline)
        grads.append(interpolated - baseline)
    return np.mean(grads, axis=0)


def shap_explanation_available() -> bool:
    """Check whether SHAP dependency is installed."""
    try:
        import shap  # noqa: F401
    except Exception:
        return False
    return True


def lime_explanation_available() -> bool:
    """Check whether LIME dependency is installed."""
    try:
        import lime  # noqa: F401
    except Exception:
        return False
    return True
