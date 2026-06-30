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
    """Generate simple pipeline architecture PNG."""

    path.parent.mkdir(parents=True, exist_ok=True)

    steps = [
        "User Query",
        "Embed Query",
        "Chroma Search",
        "Retrieve Chunks",
        "Prompt Builder",
        "Ollama Generation",
        "Final Answer",
    ]

    fig, ax = plt.subplots(figsize=(13, 2.5))
    ax.axis("off")

    x_positions = np.linspace(0.08, 0.92, len(steps))
    for idx, (x, label) in enumerate(zip(x_positions, steps, strict=False)):
        box = plt.Rectangle((x - 0.055, 0.4), 0.11, 0.2, fill=True, alpha=0.2, edgecolor="black")
        ax.add_patch(box)
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=9)
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + 0.065, 0.5),
                xytext=(x + 0.015, 0.5),
                arrowprops={"arrowstyle": "->"},
            )

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
