"""Visualization helpers for benchmarking and evaluation outputs."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import seaborn as sns

from src.constants import ARTIFACTS_DIR

_MPL_CONFIG_DIR = ARTIFACTS_DIR / ".mplconfig"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib.pyplot as plt


def plot_model_ranking(ranking: pd.DataFrame, output_path: Path) -> None:
    """Plot RMSE comparison for top models."""
    if ranking.empty:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    top = ranking.head(10).copy()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top, x="rmse", y="model_name", hue="model_name", palette="viridis", legend=False)
    plt.title("Model Comparison by RMSE (Lower Is Better)")
    plt.xlabel("RMSE")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_metric_radar(ranking: pd.DataFrame, output_path: Path) -> None:
    """Generate compact metric chart for top-ranked model."""
    if ranking.empty:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    best = ranking.iloc[0]
    labels = ["MAE", "RMSE", "MAPE", "R2"]
    values = [float(best["mae"]), float(best["rmse"]), float(best["mape"]), float(best["r2"])]
    metric_df = pd.DataFrame({"metric": labels, "value": values})

    plt.figure(figsize=(8, 4))
    sns.barplot(data=metric_df, x="metric", y="value", hue="metric", palette="mako", legend=False)
    plt.title(f"Best Model Metrics: {best['model_name']}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
