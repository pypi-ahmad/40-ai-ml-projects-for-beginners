"""Evaluation pipeline utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from domain_llm_ft.config.schemas import EvalConfig
from domain_llm_ft.evaluation.metrics import (
    classification_metrics,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from domain_llm_ft.utils.io import write_json


def run_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    config: EvalConfig,
    output_dir: Path,
) -> dict[str, float]:
    """Compute and persist evaluation artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = classification_metrics(y_true=y_true, y_pred=y_pred, y_proba=y_proba, average=config.average)

    write_json(output_dir / "metrics.json", artifacts.metrics)
    (output_dir / "classification_report.txt").write_text(artifacts.report, encoding="utf-8")
    plot_confusion_matrix(artifacts.confusion, str(output_dir / "confusion_matrix.png"))

    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        positive_scores = y_proba[:, 1]
        plot_roc_curve(y_true, positive_scores, str(output_dir / "roc_curve.png"))
        plot_pr_curve(y_true, positive_scores, str(output_dir / "pr_curve.png"))
        plot_calibration_curve(
            y_true,
            positive_scores,
            str(output_dir / "calibration_curve.png"),
            bins=config.calibration_bins,
        )

    return artifacts.metrics
