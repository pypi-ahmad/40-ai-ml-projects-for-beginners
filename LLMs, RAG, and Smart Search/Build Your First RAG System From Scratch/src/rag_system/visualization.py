"""Visualization helpers for RAG tutorial outputs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mpl_cache").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


def _prepare_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def architecture_diagram(output_path: Path) -> Path:
    """Create RAG flow diagram used in fundamentals notebook and README."""
    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axis("off")

    stages = [
        "User Query",
        "Embedding",
        "Vector Search\n(ChromaDB)",
        "Retrieved Context",
        "Prompt Builder",
        "Generation\n(qwen3.5:4b)",
        "Final Answer",
    ]
    x_positions = [0.02, 0.16, 0.30, 0.46, 0.62, 0.78, 0.92]

    for x, label in zip(x_positions, stages):
        box = FancyBboxPatch(
            (x - 0.06, 0.45),
            0.12,
            0.18,
            boxstyle="round,pad=0.02",
            edgecolor="#1f4f5f",
            facecolor="#d8eef3",
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x, 0.54, label, ha="center", va="center", fontsize=10)

    for i in range(len(x_positions) - 1):
        ax.annotate(
            "",
            xy=(x_positions[i + 1] - 0.07, 0.54),
            xytext=(x_positions[i] + 0.07, 0.54),
            arrowprops=dict(arrowstyle="->", color="#274c77", lw=1.6),
        )

    ax.set_title("Modern RAG Pipeline", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def llm_vs_rag_diagram(output_path: Path) -> Path:
    """Create side-by-side diagram: traditional LLM vs RAG."""
    _prepare_output(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    left = axes[0]
    right = axes[1]

    left.axis("off")
    right.axis("off")

    left.set_title("Traditional LLM")
    left.text(0.5, 0.75, "User Query", ha="center", bbox=dict(boxstyle="round", fc="#ffe5ec"))
    left.text(0.5, 0.5, "LLM Weights Only", ha="center", bbox=dict(boxstyle="round", fc="#ffd6a5"))
    left.text(0.5, 0.25, "Answer (hallucination risk)", ha="center", bbox=dict(boxstyle="round", fc="#ffcad4"))
    left.annotate("", xy=(0.5, 0.57), xytext=(0.5, 0.68), arrowprops=dict(arrowstyle="->"))
    left.annotate("", xy=(0.5, 0.32), xytext=(0.5, 0.43), arrowprops=dict(arrowstyle="->"))

    right.set_title("RAG System")
    right.text(0.5, 0.80, "User Query", ha="center", bbox=dict(boxstyle="round", fc="#d8f3dc"))
    right.text(0.5, 0.62, "Retriever + Vector DB", ha="center", bbox=dict(boxstyle="round", fc="#95d5b2"))
    right.text(0.5, 0.44, "Grounded Prompt", ha="center", bbox=dict(boxstyle="round", fc="#74c69d"))
    right.text(0.5, 0.26, "LLM + Citations", ha="center", bbox=dict(boxstyle="round", fc="#52b788"))
    right.annotate("", xy=(0.5, 0.69), xytext=(0.5, 0.76), arrowprops=dict(arrowstyle="->"))
    right.annotate("", xy=(0.5, 0.51), xytext=(0.5, 0.58), arrowprops=dict(arrowstyle="->"))
    right.annotate("", xy=(0.5, 0.33), xytext=(0.5, 0.40), arrowprops=dict(arrowstyle="->"))

    fig.suptitle("Why RAG Exists", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_document_lengths(doc_df: pd.DataFrame, output_path: Path) -> Path:
    """Plot distribution of document lengths."""
    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(doc_df["text"].str.len(), bins=40, kde=True, ax=ax, color="#1f77b4")
    ax.set_title("Document Length Distribution (characters)")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_chunking_comparison(results: pd.DataFrame, output_path: Path) -> Path:
    """Plot retrieval quality comparison across chunking strategies."""
    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=results, x="strategy", y="mrr", hue="strategy", ax=ax, palette="viridis", legend=False)
    ax.set_title("Chunking Strategy vs MRR")
    ax.set_xlabel("Chunking Strategy")
    ax.set_ylabel("MRR")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_retrieval_metrics(metrics: dict[str, float], output_path: Path) -> Path:
    """Plot retrieval metric summary bar chart."""
    _prepare_output(output_path)
    keys = ["precision_at_k", "recall_at_k", "f1_at_k", "mrr", "ndcg"]
    values = [float(metrics.get(key, 0.0)) for key in keys]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=keys, y=values, hue=keys, palette="mako", ax=ax, legend=False)
    ax.set_ylim(0, 1)
    ax.set_title("Retrieval Metrics")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_generation_metrics(metrics: dict[str, float], output_path: Path) -> Path:
    """Plot generation metric summary chart."""
    _prepare_output(output_path)
    keys = ["exact_match", "bleu", "rouge1", "rougeL", "meteor", "bertscore_f1"]
    values = [float(metrics.get(key, 0.0)) for key in keys]

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(x=keys, y=values, hue=keys, palette="crest", ax=ax, legend=False)
    ax.set_ylim(0, 1)
    ax.set_title("Generation Metrics")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_hallucination_comparison(df: pd.DataFrame, output_path: Path) -> Path:
    """Plot groundedness delta for RAG vs no-RAG."""
    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["groundedness_delta"], bins=30, kde=True, ax=ax, color="#2a9d8f")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Groundedness Delta: RAG - No-RAG")
    ax.set_xlabel("Delta")
    ax.set_ylabel("Query Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_table_image(df: pd.DataFrame, output_path: Path, title: str) -> Path:
    """Render a dataframe to static PNG for README-friendly reports."""
    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(10, min(6, max(2, 0.4 * len(df) + 1))))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def generate_all_core_visuals(
    doc_df: pd.DataFrame,
    retrieval_summary: dict[str, float],
    generation_summary: dict[str, float],
    hallucination_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate and save all baseline tutorial visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "rag_architecture": architecture_diagram(output_dir / "rag_architecture.png"),
        "llm_vs_rag": llm_vs_rag_diagram(output_dir / "llm_vs_rag.png"),
        "doc_lengths": plot_document_lengths(doc_df, output_dir / "doc_lengths.png"),
        "retrieval_metrics": plot_retrieval_metrics(retrieval_summary, output_dir / "retrieval_metrics.png"),
        "generation_metrics": plot_generation_metrics(generation_summary, output_dir / "generation_metrics.png"),
        "hallucination": plot_hallucination_comparison(hallucination_df, output_dir / "hallucination_delta.png"),
    }

    logger.info("Generated %d visualization artifacts", len(paths))
    return paths
