"""
visualization.py
----------------
Professional visualization utilities for the feature selection project.

All figures are stored as PNG and PDF in outputs/figures/.

Functions:
  - plot_variance_distribution  : histogram of feature variances
  - plot_correlation_heatmap    : correlation matrix heatmap
  - plot_feature_importance     : horizontal bar chart of top features
  - plot_permutation_importance : permutation importance with error bars
  - plot_rfe_results            : RFE/RFECV performance vs feature count
  - plot_mutual_information     : MI scores for top features
  - plot_shap_summary           : SHAP summary beeswarm/bar plot
  - plot_dimensionality_reduction: PCA/t-SNE/UMAP comparison
  - plot_before_after_comparison: side-by-side before/after metrics
  - plot_feature_selection_funnel: sankey-style flow of feature counts
  - plot_benchmark_comparison   : grouped bar chart of model performance
  - plot_learning_curves        : learning curves for selected models
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_IN_NOTEBOOK = "JPY_PARENT_PID" in os.environ or "IPYKERNEL_PARENT_PID" in os.environ
if not _IN_NOTEBOOK and os.environ.get("MPLBACKEND") is None:
    plt.switch_backend("Agg")

# Style setup
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("Set2")

# Output directory
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
    "figures",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_fig(name: str, dpi: int = 300, close: Optional[bool] = None):
    """Save current figure as PNG and PDF."""
    if close is None:
        close = not _IN_NOTEBOOK
    for ext in ["png", "pdf"]:
        path = os.path.join(OUTPUT_DIR, f"{name}.{ext}")
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close()


# ------------------------------------------------------------------
# A. Variance Distribution
# ------------------------------------------------------------------
def plot_variance_distribution(
    variances: pd.Series,
    threshold: float = 0.01,
    title: str = "Feature Variance Distribution",
    filename: str = "variance_distribution",
):
    """
    Histogram of feature variances with threshold line.

    Features below the threshold are candidates for removal.

    Parameters
    ----------
    variances : pd.Series
        Variance per feature.
    threshold : float
        Variance threshold line.
    title, filename : str
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(variances, bins=50, edgecolor="white", alpha=0.7)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
               label=f"Threshold = {threshold}")
    ax.set_xlabel("Variance", fontsize=12)
    ax.set_ylabel("Number of Features", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.set_yscale("log")
    _save_fig(filename)


# ------------------------------------------------------------------
# B. Correlation Heatmap
# ------------------------------------------------------------------
def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Feature Correlation Matrix",
    filename: str = "correlation_heatmap",
    figsize: Tuple[int, int] = (12, 10),
):
    """
    Heatmap of feature correlation matrix.

    For large feature sets, show a subset or use clustering.
    """
    n = corr_matrix.shape[0]
    # If too many features, show a random/representative sample
    if n > 50:
        np.random.seed(42)
        idx = np.random.choice(n, 50, replace=False)
        corr_matrix = corr_matrix.iloc[idx, idx]

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(title, fontsize=14)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    _save_fig(filename)


# ------------------------------------------------------------------
# C. Feature Importance
# ------------------------------------------------------------------
def plot_feature_importance(
    importances: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance",
    filename: str = "feature_importance",
):
    """
    Horizontal bar chart of top-N feature importances.

    Parameters
    ----------
    importances : pd.DataFrame
        Must have 'feature' and 'importance' columns.
    top_n : int
        Number of top features to show.
    """
    top = importances.sort_values("importance", ascending=False).head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top)), top["importance"].values, color="steelblue")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=9)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()
    _save_fig(filename)


# ------------------------------------------------------------------
# D. Permutation Importance
# ------------------------------------------------------------------
def plot_permutation_importance(
    imp_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Permutation Importance",
    filename: str = "permutation_importance",
):
    """
    Horizontal bar chart of permutation importance with error bars.

    Parameters
    ----------
    imp_df : pd.DataFrame
        Must have 'feature', 'importance_mean', 'importance_std' columns.
    """
    top = imp_df.sort_values("importance_mean", ascending=False).head(top_n).sort_values("importance_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        range(len(top)),
        top["importance_mean"].values,
        xerr=top["importance_std"].values,
        color="coral",
        capsize=3,
    )
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=9)
    ax.set_xlabel("Mean Importance (decrease in score)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()
    _save_fig(filename)


# ------------------------------------------------------------------
# E. RFE Results
# ------------------------------------------------------------------
def plot_rfe_results(
    cv_scores: Optional[Dict[int, float]] = None,
    n_features: Optional[List[int]] = None,
    scores: Optional[List[float]] = None,
    title: str = "RFE: Performance vs Number of Features",
    filename: str = "rfe_results",
):
    """
    Plot cross-validation score vs number of features for RFECV.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    if cv_scores:
        nf = list(cv_scores.keys())
        sc = list(cv_scores.values())
    else:
        nf = n_features or []
        sc = scores or []

    ax.plot(nf, sc, marker="o", linewidth=2, markersize=6)
    ax.set_xlabel("Number of Features", fontsize=12)
    ax.set_ylabel("Cross-Validation Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    _save_fig(filename)


# ------------------------------------------------------------------
# F. Mutual Information
# ------------------------------------------------------------------
def plot_mutual_information(
    mi_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Top Features by Mutual Information",
    filename: str = "mutual_information",
):
    """
    Horizontal bar chart of top features by mutual information.
    """
    top = mi_df.sort_values("mutual_information", ascending=False).head(top_n).sort_values("mutual_information", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top)), top["mutual_information"].values, color="seagreen")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=9)
    ax.set_xlabel("Mutual Information", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()
    _save_fig(filename)


# ------------------------------------------------------------------
# G. SHAP Summary Bar
# ------------------------------------------------------------------
def plot_shap_bar(
    shap_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Mean |SHAP| by Feature",
    filename: str = "shap_importance",
):
    """
    Bar chart of mean absolute SHAP values.
    """
    top = shap_df.sort_values("mean_abs_SHAP", ascending=False).head(top_n).sort_values("mean_abs_SHAP", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top)), top["mean_abs_SHAP"].values, color="purple")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=9)
    ax.set_xlabel("Mean |SHAP|", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()
    _save_fig(filename)


# ------------------------------------------------------------------
# H. Dimensionality Reduction Comparison
# ------------------------------------------------------------------
def plot_dimensionality_reduction(
    X_original: pd.DataFrame,
    y: pd.Series,
    X_selected: Optional[pd.DataFrame] = None,
    title: str = "PCA Visualization",
    filename: str = "pca_visualization",
):
    """
    Compare PCA of original vs selected features.
    Visualize with 2D scatter plot of first 2 principal components.
    """
    from sklearn.decomposition import PCA

    def _plot_pca(X, y, ax, title):
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis",
            alpha=0.6, s=10, edgecolor=None,
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        return scatter

    n_plots = 2 if X_selected is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(12, 5))
    if n_plots == 1:
        axes = [axes]

    _plot_pca(X_original, y, axes[0], f"Original ({X_original.shape[1]} features)")
    if X_selected is not None:
        sc = _plot_pca(X_selected, y, axes[1], f"Selected ({X_selected.shape[1]} features)")
    else:
        sc = axes[0].collections[0]

    fig.colorbar(sc, ax=axes, label="Class")
    fig.suptitle(title, fontsize=14)
    _save_fig(filename)


# ------------------------------------------------------------------
# I. Before/After Comparison
# ------------------------------------------------------------------
def plot_before_after_comparison(
    comparison_df: pd.DataFrame,
    title: str = "Before vs After Feature Selection",
    filename: str = "before_after_comparison",
):
    """
    Grouped bar chart comparing metrics before and after feature selection.
    """
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    available = [c for c in metric_cols if c in comparison_df.columns]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(available))
    width = 0.35

    before_vals = comparison_df.iloc[0][available].values if comparison_df.shape[0] > 0 else []
    after_vals = comparison_df.iloc[1][available].values if comparison_df.shape[0] > 1 else []

    if len(before_vals) == len(available):
        ax.bar(x - width / 2, before_vals, width, label="Before FS", color="steelblue")
    if len(after_vals) == len(available):
        ax.bar(x + width / 2, after_vals, width, label="After FS", color="coral")

    ax.set_xticks(x)
    ax.set_xticklabels(available, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.set_ylim(0, 1)
    _save_fig(filename)


# ------------------------------------------------------------------
# J. Feature Selection Funnel
# ------------------------------------------------------------------
def plot_feature_selection_funnel(
    stage_counts: Dict[str, int],
    title: str = "Feature Selection Funnel",
    filename: str = "feature_selection_funnel",
):
    """
    Funnel/waterfall chart showing feature count at each stage.
    """
    stages = list(stage_counts.keys())
    counts = list(stage_counts.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(stages)))

    ax.barh(range(len(stages)), counts, color=colors)
    for i, (stage, count) in enumerate(zip(stages, counts)):
        ax.text(
            count + max(counts) * 0.01, i, f"{count}",
            va="center", fontsize=11,
        )
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages, fontsize=10)
    ax.set_xlabel("Number of Features", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()
    _save_fig(filename)


# ------------------------------------------------------------------
# K. Benchmark Comparison
# ------------------------------------------------------------------
def plot_benchmark_comparison(
    results: pd.DataFrame,
    metric: str = "accuracy",
    title: str = "Model Benchmark Comparison",
    filename: str = "benchmark_comparison",
):
    """
    Grouped bar chart comparing multiple models on a single metric.
    """
    if metric not in results.columns:
        available = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]
                     if c in results.columns]
        metric = available[0] if available else results.columns[0]

    fig, ax = plt.subplots(figsize=(12, 6))
    models = results.index.tolist()
    values = results[metric].values

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    bars = ax.bar(range(len(models)), values, color=colors)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(title, fontsize=14)
    _save_fig(filename)


# ------------------------------------------------------------------
# L. Learning Curves
# ------------------------------------------------------------------
def plot_learning_curve(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
    title: str = "Learning Curve",
    filename: str = "learning_curve",
):
    """
    Plot learning curve showing train/test scores vs training examples.
    """
    from sklearn.model_selection import learning_curve

    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes,
        n_jobs=-1,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color="steelblue")
    ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std,
                    alpha=0.1, color="coral")
    ax.plot(train_sizes_abs, train_mean, "o-", label="Train score", color="steelblue")
    ax.plot(train_sizes_abs, test_mean, "o-", label="Cross-val score", color="coral")
    ax.set_xlabel("Training Examples", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(filename)
