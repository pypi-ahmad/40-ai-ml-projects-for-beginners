"""Generate publication-quality figures from benchmark outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )


def build_model_performance_chart(benchmarks_dir: Path, figures_dir: Path) -> Path:
    frame = pd.read_csv(benchmarks_dir / "model_benchmark.csv")
    successful = frame[frame["status"] == "ok"].sort_values("f1_macro", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(successful))
    ax.bar(x, successful["f1_macro"], width=0.4, label="F1 Macro", color="#0B7285")
    ax.bar(
        [idx + 0.4 for idx in x],
        successful["accuracy"],
        width=0.4,
        label="Accuracy",
        color="#74C0FC",
    )
    ax.set_xticks([idx + 0.2 for idx in x])
    ax.set_xticklabels(successful["model_name"], rotation=40, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Benchmark Comparison on Iris Test Split")
    ax.legend(loc="lower right")
    fig.tight_layout()

    output = figures_dir / "model_benchmark_scores.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def build_latency_chart(benchmarks_dir: Path, figures_dir: Path) -> Path:
    frame = pd.read_csv(benchmarks_dir / "model_benchmark.csv")
    successful = frame[frame["status"] == "ok"].sort_values("predict_time_ms", ascending=True)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.barh(successful["model_name"], successful["predict_time_ms"], color="#12B886")
    ax.set_xlabel("Prediction Time (ms)")
    ax.set_ylabel("Model")
    ax.set_title("Inference Latency by Model")
    fig.tight_layout()

    output = figures_dir / "model_inference_latency.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def build_serialization_chart(benchmarks_dir: Path, figures_dir: Path) -> Path:
    with (benchmarks_dir / "serialization_benchmark.json").open(encoding="utf-8") as handle:
        rows = json.load(handle)
    frame = pd.DataFrame(rows)
    successful = frame[frame["status"] == "ok"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(successful["format"], successful["save_time_ms"], label="Save", color="#495057")
    axes[0].bar(
        successful["format"],
        successful["load_time_ms"],
        bottom=successful["save_time_ms"],
        label="Load",
        color="#ADB5BD",
    )
    axes[0].set_title("Serialization Save/Load Time")
    axes[0].set_ylabel("Milliseconds")
    axes[0].legend()

    axes[1].bar(successful["format"], successful["size_bytes"] / 1024, color="#228BE6")
    axes[1].set_title("Artifact Size by Format")
    axes[1].set_ylabel("KB")

    fig.tight_layout()
    output = figures_dir / "serialization_benchmark.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def build_automl_chart(benchmarks_dir: Path, figures_dir: Path) -> Path | None:
    path = benchmarks_dir / "automl_benchmark.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    frame = pd.DataFrame(rows)
    successful = frame[frame["status"] == "ok"]
    if successful.empty:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(successful["framework"], successful["metric_value"], color="#1C7ED6")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Top Metric Value")
    ax.set_title("AutoML Framework Top Result Comparison")
    fig.tight_layout()

    output = figures_dir / "automl_framework_comparison.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def build_packaging_architecture(figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")
    blocks = ["Client", "API Layer", "Wrapper Layer", "Model Artifact", "Prediction"]
    xpos = [0.08, 0.28, 0.50, 0.72, 0.90]
    for label, x in zip(blocks, xpos):
        ax.text(
            x,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#FFF3BF", "edgecolor": "#F08C00"},
            transform=ax.transAxes,
        )
    for start, end in zip(xpos[:-1], xpos[1:]):
        ax.annotate(
            "",
            xy=(end - 0.06, 0.55),
            xytext=(start + 0.06, 0.55),
            arrowprops={"arrowstyle": "->", "color": "#495057", "lw": 1.8},
            xycoords=ax.transAxes,
        )
    ax.set_title("Production Packaging Architecture", fontsize=13, pad=10)
    output = figures_dir / "packaging_architecture.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def build_api_flow(figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10.2, 3.0))
    ax.axis("off")
    blocks = ["POST /predict", "Validate (Pydantic)", "Inference Engine", "Response + Metrics Log"]
    xpos = [0.10, 0.35, 0.60, 0.86]
    for label, x in zip(blocks, xpos):
        ax.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#E7F5FF", "edgecolor": "#1C7ED6"},
            transform=ax.transAxes,
        )
    for start, end in zip(xpos[:-1], xpos[1:]):
        ax.annotate(
            "",
            xy=(end - 0.075, 0.5),
            xytext=(start + 0.075, 0.5),
            arrowprops={"arrowstyle": "->", "color": "#1864AB", "lw": 1.8},
            xycoords=ax.transAxes,
        )
    ax.set_title("FastAPI Prediction Flow", fontsize=13, pad=10)
    output = figures_dir / "api_prediction_flow.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def build_versioning_workflow(figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")

    steps = ["Train", "Benchmark", "Serialize", "Register", "Activate", "Serve"]
    x_positions = [0.06, 0.23, 0.40, 0.57, 0.74, 0.90]

    for step, xpos in zip(steps, x_positions):
        ax.text(
            xpos,
            0.55,
            step,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F1F3F5", "edgecolor": "#868E96"},
            transform=ax.transAxes,
        )

    for start, end in zip(x_positions[:-1], x_positions[1:]):
        ax.annotate(
            "",
            xy=(end - 0.05, 0.55),
            xytext=(start + 0.05, 0.55),
            arrowprops={"arrowstyle": "->", "color": "#495057", "lw": 1.8},
            xycoords=ax.transAxes,
        )

    ax.set_title("Model Packaging and Deployment Lifecycle", fontsize=13, pad=10)
    output = figures_dir / "packaging_lifecycle_workflow.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    _style()
    benchmarks_dir = Path("outputs/benchmarks")
    figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not benchmarks_dir.exists():
        raise FileNotFoundError(
            "Missing outputs/benchmarks. Run scripts/train_model.py first."
        )

    created = [
        build_model_performance_chart(benchmarks_dir, figures_dir),
        build_latency_chart(benchmarks_dir, figures_dir),
        build_serialization_chart(benchmarks_dir, figures_dir),
        build_packaging_architecture(figures_dir),
        build_api_flow(figures_dir),
        build_versioning_workflow(figures_dir),
    ]
    automl_chart = build_automl_chart(benchmarks_dir, figures_dir)
    if automl_chart is not None:
        created.append(automl_chart)
    for artifact in created:
        print(f"Created figure: {artifact}")


if __name__ == "__main__":
    main()
