"""Visualization utilities for benchmark metrics and workflow diagrams."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.schemas import BenchmarkResult

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


class BenchmarkVisualizer:
    """Generate production-ready static figures for app, notebooks, and README."""

    def __init__(self, output_dir: str = "outputs/figures") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.grid": True,
                "grid.alpha": 0.25,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "font.size": 11,
            }
        )

    @staticmethod
    def _normalize(results: list[BenchmarkResult] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize typed objects into plain dictionaries."""

        return [
            item.model_dump() if isinstance(item, BenchmarkResult) else item for item in results
        ]

    def latency_bar(self, results: list[BenchmarkResult] | list[dict[str, Any]]) -> str:
        """Create horizontal latency chart."""

        records = [row for row in self._normalize(results) if not row.get("error")]
        models = [row["model"] for row in records]
        values = [row["mean_latency_ms"] for row in records]

        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        colors = plt.cm.Blues(np.linspace(0.45, 0.9, len(models)))
        bars = ax.barh(models, values, color=colors)

        for bar, value in zip(bars, values, strict=False):
            ax.text(value + 4, bar.get_y() + bar.get_height() / 2, f"{value:.1f} ms", va="center")

        ax.set_title("Model Latency Comparison")
        ax.set_xlabel("Mean Latency (ms)")
        ax.set_xlim(0, max(values) * 1.2 if values else 1)
        path = self.output_dir / "latency_comparison.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def throughput(self, results: list[BenchmarkResult] | list[dict[str, Any]]) -> str:
        """Create tokens-per-second throughput chart."""

        records = [row for row in self._normalize(results) if not row.get("error")]
        models = [row["model"] for row in records]
        values = [row["mean_tokens_per_sec"] for row in records]

        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        colors = plt.cm.Greens(np.linspace(0.45, 0.9, len(models)))
        bars = ax.bar(models, values, color=colors)

        for bar, value in zip(bars, values, strict=False):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.1f}", ha="center")

        ax.set_title("Model Throughput Comparison")
        ax.set_ylabel("Tokens / second")
        ax.tick_params(axis="x", rotation=15)
        path = self.output_dir / "throughput_comparison.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def memory_usage(self, results: list[BenchmarkResult] | list[dict[str, Any]]) -> str:
        """Create memory overhead comparison chart."""

        records = [row for row in self._normalize(results) if not row.get("error")]
        models = [row["model"] for row in records]
        values = [row.get("mean_memory_mb", 0.0) for row in records]

        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        bars = ax.bar(models, values, color="#F4A261")
        for bar, value in zip(bars, values, strict=False):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.2f}", ha="center")

        ax.set_title("Incremental Memory Usage During Inference")
        ax.set_ylabel("Memory Delta (MB)")
        ax.tick_params(axis="x", rotation=15)
        path = self.output_dir / "memory_usage_comparison.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def prompt_scale(
        self,
        short_results: list[BenchmarkResult] | list[dict[str, Any]],
        medium_results: list[BenchmarkResult] | list[dict[str, Any]],
        long_results: list[BenchmarkResult] | list[dict[str, Any]],
    ) -> str:
        """Create grouped chart showing latency scaling across prompt sizes."""

        short_map = {
            item["model"]: item["mean_latency_ms"]
            for item in self._normalize(short_results)
            if not item.get("error")
        }
        medium_map = {
            item["model"]: item["mean_latency_ms"]
            for item in self._normalize(medium_results)
            if not item.get("error")
        }
        long_map = {
            item["model"]: item["mean_latency_ms"]
            for item in self._normalize(long_results)
            if not item.get("error")
        }

        models = sorted(set(short_map) | set(medium_map) | set(long_map))
        x_axis = np.arange(len(models))
        width = 0.27

        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        ax.bar(
            x_axis - width,
            [short_map.get(model, 0.0) for model in models],
            width,
            label="Short",
            color="#5DADE2",
        )
        ax.bar(
            x_axis,
            [medium_map.get(model, 0.0) for model in models],
            width,
            label="Medium",
            color="#F39C12",
        )
        ax.bar(
            x_axis + width,
            [long_map.get(model, 0.0) for model in models],
            width,
            label="Long",
            color="#E74C3C",
        )

        ax.set_title("Latency Scaling by Prompt Complexity")
        ax.set_ylabel("Mean Latency (ms)")
        ax.set_xticks(x_axis)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.legend()
        path = self.output_dir / "prompt_scale_comparison.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def radar(self, results: list[BenchmarkResult] | list[dict[str, Any]]) -> str:
        """Create radar chart summarizing speed/throughput/memory/quality tradeoffs."""

        records = [row for row in self._normalize(results) if not row.get("error")]
        if not records:
            return ""

        labels = ["Speed", "Throughput", "Memory Efficiency", "Quality"]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        max_latency = max(row["mean_latency_ms"] for row in records)
        max_tps = max(row["mean_tokens_per_sec"] for row in records)
        max_memory = max(row.get("mean_memory_mb", 0.0) for row in records) or 1.0

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True}, constrained_layout=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(records)))

        for row, color in zip(records, colors, strict=False):
            speed = 100 - (row["mean_latency_ms"] / max_latency * 100)
            throughput_score = row["mean_tokens_per_sec"] / max_tps * 100
            memory_efficiency = 100 - (row.get("mean_memory_mb", 0.0) / max_memory * 100)
            quality_score = row.get("quality_score", 0.0)

            values = [speed, throughput_score, memory_efficiency, quality_score]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=row["model"], color=color)
            ax.fill(angles, values, alpha=0.08, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)
        ax.set_title("Model Tradeoff Radar")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        path = self.output_dir / "model_radar.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def workflow_diagram(self) -> str:
        """Generate layered ML application architecture diagram."""

        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        ax.axis("off")

        layers = [
            ("User Interface Layer", 0.06, "#E3F2FD"),
            ("Inference Orchestration Layer", 0.24, "#E8F5E9"),
            ("Model Layer (Ollama)", 0.42, "#FFF3E0"),
            ("Data / File Layer", 0.60, "#F3E5F5"),
            ("Deployment Layer", 0.78, "#FBE9E7"),
        ]

        for label, x_position, color in layers:
            rectangle = plt.Rectangle(
                (x_position, 0.35), 0.16, 0.3, color=color, ec="#37474F", lw=1.5
            )
            ax.add_patch(rectangle)
            ax.text(
                x_position + 0.08, 0.5, label, ha="center", va="center", fontsize=10, weight="bold"
            )

        for start in [0.22, 0.40, 0.58, 0.76]:
            ax.annotate(
                "",
                xy=(start + 0.02, 0.5),
                xytext=(start, 0.5),
                arrowprops={"arrowstyle": "->", "lw": 2},
            )

        ax.text(
            0.5,
            0.16,
            "Request Flow: Input -> Preprocess -> Inference -> Postprocess -> Render",
            ha="center",
            fontsize=11,
        )

        path = self.output_dir / "ml_app_architecture.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def generate_all(
        self,
        medium_results: list[BenchmarkResult] | list[dict[str, Any]],
        short_results: list[BenchmarkResult] | list[dict[str, Any]] | None = None,
        long_results: list[BenchmarkResult] | list[dict[str, Any]] | None = None,
    ) -> dict[str, str]:
        """Generate all standard project figures in one call."""

        artifacts = {
            "latency": self.latency_bar(medium_results),
            "throughput": self.throughput(medium_results),
            "memory": self.memory_usage(medium_results),
            "radar": self.radar(medium_results),
            "workflow": self.workflow_diagram(),
        }

        if short_results is not None and long_results is not None:
            artifacts["prompt_scale"] = self.prompt_scale(
                short_results, medium_results, long_results
            )

        logger.info("Generated figure artifacts: %s", artifacts)
        return artifacts
