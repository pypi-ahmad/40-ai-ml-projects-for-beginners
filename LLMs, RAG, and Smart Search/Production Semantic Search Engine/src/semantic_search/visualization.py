"""Plotly visualization utilities for analytics and benchmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def embedding_projection_plot(
    embeddings: np.ndarray,
    labels: list[str] | None = None,
    method: str = "pca",
):
    """Create 2D projection plot of embeddings."""
    if embeddings.ndim != 2:
        raise ValueError("Expected 2D embedding matrix")

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2, random_state=42)

    points = reducer.fit_transform(embeddings)
    fig = px.scatter(
        x=points[:, 0],
        y=points[:, 1],
        color=labels if labels else None,
        title=f"Embedding Projection ({method.upper()})",
        labels={"x": "Component 1", "y": "Component 2"},
    )
    return fig


def similarity_histogram(similarities: list[float]):
    """Plot histogram of similarity scores."""
    return px.histogram(similarities, nbins=40, title="Similarity Distribution")


def latency_line_chart(latencies: list[float], title: str = "Query Latency"):
    """Plot latency trend line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=latencies, mode="lines+markers", name="latency_ms"))
    fig.update_layout(title=title, xaxis_title="Query #", yaxis_title="Latency (ms)")
    return fig


def precision_recall_curve(precision: list[float], recall: list[float], title: str = "Precision-Recall"):
    """Plot precision-recall curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines+markers", name="PR Curve"))
    fig.update_layout(title=title, xaxis_title="Recall", yaxis_title="Precision")
    return fig


def ranking_comparison_chart(labels: list[str], values: list[float], title: str = "Ranking Comparison"):
    """Plot bar chart for ranking metric comparisons."""
    return px.bar(x=labels, y=values, title=title, labels={"x": "System", "y": "Score"})


def save_plot(fig, output_path: str | Path) -> None:
    """Write plot as HTML artifact."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(target), include_plotlyjs="cdn")
