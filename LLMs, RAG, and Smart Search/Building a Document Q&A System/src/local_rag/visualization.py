"""Visualization utilities for embeddings, metrics, and architecture artifacts."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
except ImportError:  # pragma: no cover - optional dependency
    umap = None


def project_embeddings(embeddings: np.ndarray, method: str = "pca") -> np.ndarray:
    """Project high-dimensional vectors into 2D."""

    method = method.lower()
    if method == "pca":
        return PCA(n_components=2, random_state=42).fit_transform(embeddings)
    if method == "tsne":
        projector = TSNE(
            n_components=2,
            random_state=42,
            init="pca",
            learning_rate="auto",
        )
        return projector.fit_transform(embeddings)
    if method == "umap":
        if umap is None:
            raise ImportError("umap-learn is not installed. Install it to use UMAP projection.")
        return umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
    raise ValueError(f"Unsupported projection method: {method}")


def embedding_projection_figure(embeddings: np.ndarray, labels: list[str], method: str = "pca"):
    """Create plotly scatter for embedding projection."""

    points = project_embeddings(embeddings, method=method)
    fig = px.scatter(
        x=points[:, 0],
        y=points[:, 1],
        color=labels,
        title=f"Embedding Projection ({method.upper()})",
    )
    fig.update_layout(xaxis_title="dim_1", yaxis_title="dim_2")
    return fig



def save_pipeline_architecture_diagram(path: Path) -> None:
    """Generate pipeline architecture PNG."""

    path.parent.mkdir(parents=True, exist_ok=True)

    steps = [
        "User Question",
        "Query Embedding",
        "Retriever",
        "Hybrid Rank",
        "Context Builder",
        "Prompt Template",
        "LLM Generation",
        "Citations",
        "Evaluation",
    ]

    fig, ax = plt.subplots(figsize=(16, 2.8))
    ax.axis("off")

    x_positions = np.linspace(0.05, 0.95, len(steps))
    for idx, (x, label) in enumerate(zip(x_positions, steps, strict=False)):
        box = plt.Rectangle((x - 0.05, 0.4), 0.1, 0.24, fill=True, alpha=0.2, edgecolor="black")
        ax.add_patch(box)
        ax.text(x, 0.52, label, ha="center", va="center", fontsize=9)
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + 0.06, 0.52),
                xytext=(x + 0.015, 0.52),
                arrowprops={"arrowstyle": "->"},
            )

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)



def save_chunk_boundary_plot(path: Path, text: str, chunk_size: int, overlap: int) -> None:
    """Visualize chunk boundaries over text length."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text_len = len(text)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < text_len:
        end = min(start + chunk_size, text_len)
        spans.append((start, end))
        if end >= text_len:
            break
        start = max(0, end - overlap)

    fig, ax = plt.subplots(figsize=(12, 1 + len(spans) * 0.28))
    for idx, (start_idx, end_idx) in enumerate(spans):
        ax.barh(idx, end_idx - start_idx, left=start_idx, height=0.5, color="#6BAED6")
    ax.set_xlabel("Character position")
    ax.set_ylabel("Chunk index")
    ax.set_title(f"Chunk boundaries (size={chunk_size}, overlap={overlap})")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)



def save_latency_bar(path: Path, labels: list[str], values_ms: list[float]) -> None:
    """Save latency bar chart."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, values_ms, color="#74C476")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
