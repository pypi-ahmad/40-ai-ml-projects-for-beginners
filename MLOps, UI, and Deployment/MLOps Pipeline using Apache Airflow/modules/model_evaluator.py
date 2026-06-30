"""Model evaluation metrics and diagnostic plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


def _save_plot(fig: plt.Figure, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def save_diagnostic_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path,
    prefix: str,
) -> dict[str, str]:
    """Save residual, prediction-vs-actual, and error distribution plots."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(0.0, color="red", linestyle="--", linewidth=1)
    ax1.set_title("Residuals vs Predicted")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Residual")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.scatter(y_true, y_pred, alpha=0.6)
    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
    ax2.set_title("Prediction vs Actual")
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.hist(residuals, bins=30, alpha=0.8)
    ax3.set_title("Error Distribution")
    ax3.set_xlabel("Residual")
    ax3.set_ylabel("Frequency")

    return {
        "residual_plot": _save_plot(fig1, out_dir / f"{prefix}_residuals.png"),
        "prediction_plot": _save_plot(fig2, out_dir / f"{prefix}_prediction_vs_actual.png"),
        "error_dist_plot": _save_plot(fig3, out_dir / f"{prefix}_error_distribution.png"),
    }


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figures_dir: str | Path,
    prefix: str = "model",
) -> dict[str, Any]:
    """Evaluate trained model and generate diagnostics."""
    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_true=y_test.to_numpy(), y_pred=np.asarray(y_pred))
    figures = save_diagnostic_plots(y_true=y_test.to_numpy(), y_pred=np.asarray(y_pred), output_dir=figures_dir, prefix=prefix)
    return {"metrics": metrics, "figures": figures}
