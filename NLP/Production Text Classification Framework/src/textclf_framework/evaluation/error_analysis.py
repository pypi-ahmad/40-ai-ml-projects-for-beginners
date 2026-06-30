"""Error analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


@dataclass(slots=True)
class ErrorSlice:
    text: str
    true_label: int
    pred_label: int
    confidence: float
    entropy: float


def prediction_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute entropy per prediction distribution."""
    safe_probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(safe_probs * np.log(safe_probs), axis=1)


def confusion_matrix_df(labels: list[int], preds: list[int], label_names: list[str]) -> pd.DataFrame:
    """Create confusion matrix as labeled dataframe."""
    cm = confusion_matrix(labels, preds, labels=list(range(len(label_names))))
    return pd.DataFrame(cm, index=label_names, columns=label_names)


def misclassified_samples(
    texts: list[str],
    labels: list[int],
    preds: list[int],
    probs: np.ndarray,
    max_rows: int = 100,
) -> pd.DataFrame:
    """Return dataframe of misclassified examples sorted by confidence."""
    entropies = prediction_entropy(probs)
    rows: list[dict[str, object]] = []

    for idx, (truth, pred) in enumerate(zip(labels, preds, strict=True)):
        if truth == pred:
            continue
        confidence = float(np.max(probs[idx]))
        rows.append(
            {
                "index": idx,
                "text": texts[idx],
                "true_label": int(truth),
                "pred_label": int(pred),
                "confidence": confidence,
                "entropy": float(entropies[idx]),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["confidence", "entropy"], ascending=[False, False]).head(max_rows)
