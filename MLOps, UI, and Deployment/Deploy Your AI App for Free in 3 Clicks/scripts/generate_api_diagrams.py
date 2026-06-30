"""Generate API architecture and request flow diagrams."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _box(ax, x, y, w, h, text, color):
    rect = patches.FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.05",
        linewidth=1.5,
        edgecolor="#0f172a",
        facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=9, color="#0f172a")


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "lw": 1.6})


def architecture_diagram() -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    _box(ax, 2, 6.5, 3.0, 1.0, "Client\n(REST Consumer)", "#dbeafe")
    _box(ax, 5, 6.5, 3.0, 1.0, "FastAPI Routers\n(validation + errors)", "#fee2e2")
    _box(ax, 8, 6.5, 3.0, 1.0, "Service Layer\n(inference + explain)", "#dcfce7")
    _box(ax, 5, 4.5, 3.2, 1.0, "Model Artifacts\n(joblib/pickle)", "#fef9c3")
    _box(ax, 8.9, 4.5, 2.2, 1.0, "Metrics Store", "#ede9fe")
    _box(ax, 2.2, 4.5, 3.2, 1.0, "Training Pipeline\nbenchmark + automl", "#ffedd5")
    _box(ax, 2.2, 2.6, 3.2, 1.0, "Curated Ames CSV", "#f1f5f9")

    _arrow(ax, 3.5, 6.5, 4.0, 6.5)
    _arrow(ax, 6.5, 6.5, 7.0, 6.5)
    _arrow(ax, 8.0, 6.0, 6.0, 4.95)
    _arrow(ax, 8.8, 6.0, 8.9, 5.0)
    _arrow(ax, 2.2, 4.0, 2.2, 3.1)
    _arrow(ax, 3.8, 4.5, 4.0, 4.5)

    ax.text(6, 7.5, "FastAPI ML Serving Architecture", fontsize=14, fontweight="bold")
    fig.savefig(FIG_DIR / "fastapi-architecture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def request_flow_diagram() -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    steps = [
        (1.4, "Request"),
        (3.3, "Pydantic\nValidation"),
        (5.2, "Model\nInference"),
        (7.1, "Metrics\nUpdate"),
        (9.0, "Response\nEnvelope"),
        (10.8, "Client"),
    ]
    colors = ["#dbeafe", "#fee2e2", "#dcfce7", "#ede9fe", "#fef9c3", "#ffedd5"]

    for (x, text), color in zip(steps, colors):
        _box(ax, x, 2.5, 1.6, 0.9, text, color)

    for idx in range(len(steps) - 1):
        _arrow(ax, steps[idx][0] + 0.8, 2.5, steps[idx + 1][0] - 0.8, 2.5)

    ax.text(6, 4.2, "Prediction Request Flow", fontsize=14, fontweight="bold")
    fig.savefig(FIG_DIR / "fastapi-request-flow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    architecture_diagram()
    request_flow_diagram()
    print("Generated API diagrams: fastapi-architecture.png, fastapi-request-flow.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
