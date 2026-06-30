"""Evaluation metrics and reports."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvalArtifacts:
    metrics: dict[str, float]
    report: str
    confusion: np.ndarray


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    average: str,
) -> EvalArtifacts:
    """Compute broad classification metrics."""
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    if y_proba is not None:
        try:
            if y_proba.ndim == 1:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            elif y_proba.ndim == 2 and y_proba.shape[1] > 1:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average=average))
        except ValueError:
            pass

    return EvalArtifacts(
        metrics=metrics,
        report=classification_report(y_true, y_pred, zero_division=0),
        confusion=confusion_matrix(y_true, y_pred),
    )


def plot_confusion_matrix(confusion: np.ndarray, output_path: str) -> None:
    """Render confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    plt.imshow(confusion, cmap="Blues")
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(j, i, int(confusion[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, output_path: str) -> None:
    """Plot ROC curve for binary classification."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, output_path: str) -> None:
    """Plot precision-recall curve for binary classification."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_calibration_curve(y_true: np.ndarray, y_score: np.ndarray, output_path: str, bins: int) -> None:
    """Plot calibration curve for binary classification."""
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=bins)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("Calibration Curve")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
