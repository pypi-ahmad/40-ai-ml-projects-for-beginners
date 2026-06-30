"""Explainability utilities: attention, SHAP, LIME, integrated gradients."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from domain_llm_ft.utils.io import write_json


def save_attention_map(attentions: np.ndarray, output_path: Path) -> None:
    """Save mean attention heatmap for first head/layer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = attentions.mean(axis=0)
    plt.figure(figsize=(6, 5))
    plt.imshow(arr, cmap="viridis")
    plt.title("Attention Heatmap")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def export_token_importance(tokens: list[str], scores: np.ndarray, output_path: Path) -> None:
    """Write token importance payload."""
    payload = {token: float(score) for token, score in zip(tokens, scores, strict=False)}
    write_json(output_path, payload)


def summarize_confidence(probabilities: np.ndarray) -> dict[str, float]:
    """Aggregate prediction confidence summary."""
    return {
        "mean_confidence": float(np.max(probabilities, axis=1).mean()),
        "min_confidence": float(np.max(probabilities, axis=1).min()),
        "max_confidence": float(np.max(probabilities, axis=1).max()),
    }
