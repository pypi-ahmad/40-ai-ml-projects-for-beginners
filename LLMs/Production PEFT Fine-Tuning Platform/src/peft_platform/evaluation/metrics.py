"""Evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EvalResult:
    accuracy: float
    f1: float
    exact_match: float


def evaluate_classification(y_true: list[int], y_pred: list[int]) -> EvalResult:
    if not y_true:
        return EvalResult(accuracy=0.0, f1=0.0, exact_match=0.0)

    matches = [int(a == b) for a, b in zip(y_true, y_pred, strict=True)]
    acc = float(sum(matches) / len(matches))

    labels = sorted(set(y_true) | set(y_pred))
    weighted_f1 = 0.0
    total = len(y_true)
    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt == label and yp != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_label = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = sum(1 for yt in y_true if yt == label)
        weighted_f1 += f1_label * (support / total)
    f1 = float(weighted_f1)
    exact = float(sum(int(a == b) for a, b in zip(y_true, y_pred, strict=True)) / len(y_true))
    return EvalResult(accuracy=acc, f1=f1, exact_match=exact)
