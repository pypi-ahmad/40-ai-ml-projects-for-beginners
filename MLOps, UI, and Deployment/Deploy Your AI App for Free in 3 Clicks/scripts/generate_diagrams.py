"""Generate professional diagrams for Project #7 deployment tutorial.

Outputs PNG artifacts under ``outputs/figures`` and is safe to run in headless
CI environments because it forces the matplotlib Agg backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Palette:
    """Centralized color palette for visual consistency across diagrams."""

    bg: str = "#f8fafc"
    ink: str = "#0f172a"
    sub_ink: str = "#334155"
    orange: str = "#fb923c"
    blue: str = "#60a5fa"
    green: str = "#4ade80"
    violet: str = "#a78bfa"
    rose: str = "#fb7185"
    amber: str = "#facc15"


PALETTE = Palette()


def _style_axis(ax):
    ax.set_facecolor(PALETTE.bg)
    ax.axis("off")


def _add_round_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    face: str,
    edge: str,
    fontsize: int = 10,
    weight: str = "bold",
):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        linewidth=1.8,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=PALETTE.ink, fontweight=weight)


def _add_arrow(ax, x1: float, y1: float, x2: float, y2: float, color: str = PALETTE.sub_ink):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": color},
    )


def draw_deployment_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    _style_axis(ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    _add_round_box(ax, 2.2, 6.7, 3.2, 1.0, "Developer Machine\n(uv + local tests)", PALETTE.orange, "#c2410c")
    _add_round_box(ax, 6.0, 6.7, 3.2, 1.0, "GitHub Repository\n(source + workflow)", PALETTE.blue, "#1d4ed8")
    _add_round_box(ax, 9.8, 6.7, 3.2, 1.0, "Streamlit Cloud\n(build + host)", PALETTE.green, "#166534")

    _add_round_box(ax, 9.8, 3.9, 3.2, 1.0, "Public App URL\n(user traffic)", "#dcfce7", "#15803d")
    _add_round_box(ax, 6.0, 3.9, 3.2, 1.0, "HF Inference API\n(primary model runtime)", "#fee2e2", "#be123c")
    _add_round_box(ax, 2.2, 3.9, 3.2, 1.0, "Fallback Runtime\n(Ollama / rules)", "#ede9fe", "#6d28d9")

    _add_arrow(ax, 3.8, 6.7, 4.4, 6.7)
    _add_arrow(ax, 7.6, 6.7, 8.2, 6.7)
    _add_arrow(ax, 9.8, 6.1, 9.8, 4.5)
    _add_arrow(ax, 8.3, 3.9, 7.7, 3.9)
    _add_arrow(ax, 4.0, 3.9, 4.9, 3.9)

    ax.text(6.0, 1.1, "Code push -> cloud build -> live endpoint with graceful inference fallback", fontsize=11, color=PALETTE.sub_ink, ha="center")
    ax.text(6.0, 7.7, "Deployment Architecture", fontsize=14, color=PALETTE.ink, ha="center", fontweight="bold")

    fig.savefig(FIGURES_DIR / "deployment-architecture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_application_lifecycle() -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    _style_axis(ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)

    stages = [
        (1.3, "Develop"),
        (3.3, "Test"),
        (5.3, "Package"),
        (7.3, "Deploy"),
        (9.3, "Monitor"),
        (11.1, "Maintain"),
    ]
    fills = [PALETTE.orange, PALETTE.amber, PALETTE.blue, PALETTE.green, PALETTE.violet, PALETTE.rose]
    edges = ["#c2410c", "#a16207", "#1d4ed8", "#166534", "#6d28d9", "#be123c"]

    for (x, label), fill, edge in zip(stages, fills, edges):
        _add_round_box(ax, x, 2.4, 1.5, 0.9, label, fill, edge)

    for i in range(len(stages) - 1):
        _add_arrow(ax, stages[i][0] + 0.76, 2.4, stages[i + 1][0] - 0.76, 2.4)

    ax.text(6.2, 4.2, "AI Application Lifecycle", fontsize=14, color=PALETTE.ink, ha="center", fontweight="bold")
    ax.text(6.2, 0.8, "Loop repeats after every model/app update", fontsize=11, color=PALETTE.sub_ink, ha="center")

    fig.savefig(FIGURES_DIR / "application-lifecycle.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_git_workflow() -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    _style_axis(ax)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)

    _add_round_box(ax, 1.5, 4.8, 2.3, 0.9, "Local Branch\nfeature/deploy", "#ffedd5", "#c2410c")
    _add_round_box(ax, 4.4, 4.8, 2.3, 0.9, "Commit\nsmall + clear", "#dbeafe", "#1d4ed8")
    _add_round_box(ax, 7.3, 4.8, 2.3, 0.9, "Push\norigin/main", "#dcfce7", "#15803d")
    _add_round_box(ax, 9.5, 4.8, 2.3, 0.9, "Cloud Build\ntriggered", "#ede9fe", "#6d28d9")

    _add_round_box(ax, 5.5, 2.0, 4.5, 1.2, "PR / review / CI checks\npytest + notebook execution + lint", "#fef9c3", "#a16207", fontsize=10)

    _add_arrow(ax, 2.65, 4.8, 3.25, 4.8)
    _add_arrow(ax, 5.55, 4.8, 6.15, 4.8)
    _add_arrow(ax, 8.45, 4.8, 8.95, 4.8)
    _add_arrow(ax, 7.3, 4.3, 6.1, 2.6)
    _add_arrow(ax, 3.2, 2.0, 1.8, 4.35, color="#be123c")

    ax.text(5.6, 5.6, "Git Workflow for Deployment", fontsize=14, color=PALETTE.ink, ha="center", fontweight="bold")
    ax.text(5.5, 0.8, "Short feedback loops reduce production failures", fontsize=11, color=PALETTE.sub_ink, ha="center")

    fig.savefig(FIGURES_DIR / "git-workflow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_cloud_workflow() -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    _style_axis(ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    _add_round_box(ax, 2.0, 4.2, 3.0, 1.0, "1. Connect Repo\nStreamlit Cloud", "#dbeafe", "#1d4ed8")
    _add_round_box(ax, 6.0, 4.2, 3.0, 1.0, "2. Configure App\nentrypoint + Python", "#dcfce7", "#166534")
    _add_round_box(ax, 10.0, 4.2, 3.0, 1.0, "3. Add Secrets\nHF_API_TOKEN", "#fef3c7", "#a16207")

    _add_round_box(ax, 6.0, 1.9, 4.6, 1.1, "Build logs -> health check -> public URL", "#fee2e2", "#be123c")

    _add_arrow(ax, 3.5, 4.2, 4.5, 4.2)
    _add_arrow(ax, 7.5, 4.2, 8.5, 4.2)
    _add_arrow(ax, 10.0, 3.6, 6.9, 2.5)

    ax.text(6.0, 5.4, "Streamlit Cloud Workflow", fontsize=14, color=PALETTE.ink, ha="center", fontweight="bold")

    fig.savefig(FIGURES_DIR / "cloud-workflow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_fallback_strategy() -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    _style_axis(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    _add_round_box(ax, 5.0, 5.0, 5.0, 0.9, "Incoming request", "#ffedd5", "#c2410c")
    _add_round_box(ax, 5.0, 3.8, 6.8, 0.9, "Tier 1: Hugging Face Inference API", "#fee2e2", "#be123c")
    _add_round_box(ax, 5.0, 2.5, 6.8, 0.9, "Tier 2: Ollama local model", "#ede9fe", "#6d28d9")
    _add_round_box(ax, 5.0, 1.2, 6.8, 0.9, "Tier 3: Rule-based fallback", "#dcfce7", "#166534")

    _add_arrow(ax, 5.0, 4.55, 5.0, 4.2)
    _add_arrow(ax, 5.0, 3.35, 5.0, 2.9)
    _add_arrow(ax, 5.0, 2.05, 5.0, 1.6)

    ax.text(5.0, 0.35, "Goal: always return useful output even during provider failure", fontsize=10.5, color=PALETTE.sub_ink, ha="center")
    ax.text(5.0, 5.65, "Inference Fallback Strategy", fontsize=14, color=PALETTE.ink, ha="center", fontweight="bold")

    fig.savefig(FIGURES_DIR / "fallback-strategy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    draw_deployment_architecture()
    draw_application_lifecycle()
    draw_git_workflow()
    draw_cloud_workflow()
    draw_fallback_strategy()

    generated = sorted(path.name for path in FIGURES_DIR.glob("*.png"))
    print("Generated diagrams:")
    for name in generated:
        print(f"- {name}")


if __name__ == "__main__":
    main()
