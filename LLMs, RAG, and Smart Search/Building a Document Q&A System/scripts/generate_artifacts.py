"""Generate static diagram artifacts for README and reports."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from local_rag.config import AppSettings
from local_rag.visualization import save_pipeline_architecture_diagram


def _latency_chart(path: Path) -> None:
    labels = ["Embedding", "Retrieval", "Generation"]
    values = [120, 45, 280]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_title("Sample Latency Breakdown")
    ax.set_ylabel("ms")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _similarity_chart(path: Path) -> None:
    x = np.arange(1, 11)
    y = np.exp(-x / 4)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o")
    ax.set_title("Example Similarity Score Curve")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Cosine Similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    settings = AppSettings()
    settings.ensure_directories()

    save_pipeline_architecture_diagram(settings.diagrams_dir / "pipeline_architecture.png")
    _latency_chart(settings.diagrams_dir / "latency_breakdown.png")
    _similarity_chart(settings.diagrams_dir / "similarity_curve.png")

    screenshot_dir = settings.outputs_dir / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        (
            "Streamlit UI Screenshot Placeholder\\n"
            "Run `uv run streamlit run streamlit_app/app.py`\\n"
            "Then capture real screens."
        ),
        ha="center",
        va="center",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(screenshot_dir / "streamlit_placeholder.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
