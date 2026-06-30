"""Evaluation metrics for classification tasks."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from transformers import EvalPrediction


def compute_classification_metrics(
    labels: Sequence[int],
    preds: Sequence[int],
    probs: np.ndarray | None = None,
    top_k: tuple[int, ...] = (3,),
) -> dict[str, float]:
    """Compute broad metric suite for text classification."""
    y_true = np.asarray(labels)
    y_pred = np.asarray(preds)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
    }

    if probs is not None:
        try:
            metrics["log_loss"] = float(log_loss(y_true, probs))
        except ValueError:
            metrics["log_loss"] = float("nan")

        n_classes = probs.shape[1]
        for k in top_k:
            if k <= n_classes:
                metrics[f"top_{k}_accuracy"] = float(top_k_accuracy_score(y_true, probs, k=k))

        unique = np.unique(y_true)
        if len(unique) == 2 and probs.shape[1] >= 2:
            positive = probs[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_true, positive))
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, positive)
            metrics["pr_auc"] = float(np.trapezoid(precision_curve, recall_curve))

    return metrics


def compute_metrics_for_trainer(eval_pred: EvalPrediction) -> dict[str, float]:
    """Hugging Face Trainer-compatible metric callback."""
    logits, labels = eval_pred
    probs = softmax(logits, axis=1)
    preds = np.argmax(probs, axis=1)
    return compute_classification_metrics(labels=labels, preds=preds, probs=probs)
