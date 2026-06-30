"""Plotly visualizations for benchmark and runtime analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.express as px
import plotly.graph_objects as go


class BenchmarkVisualizer:
    """Generate benchmark plots as standalone HTML files."""

    def __init__(self, output_dir: str | Path = "artifacts/plots") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def latency_chart(self, summary: list[dict[str, Any]]) -> Path:
        """Create latency bar chart."""

        fig = px.bar(summary, x="model", y="avg_latency_ms", title="Average Latency by Model")
        target = self.output_dir / "latency_by_model.html"
        fig.write_html(str(target), include_plotlyjs="cdn")
        return target

    def accuracy_chart(self, summary: list[dict[str, Any]]) -> Path:
        """Create hybrid accuracy chart."""

        fig = px.bar(summary, x="model", y="accuracy", title="Hybrid Accuracy by Model")
        target = self.output_dir / "accuracy_by_model.html"
        fig.write_html(str(target), include_plotlyjs="cdn")
        return target

    def radar_chart(self, summary: list[dict[str, Any]]) -> Path:
        """Create radar chart for selected metrics."""

        metrics = ["accuracy", "success_rate", "avg_tool_calls"]
        fig = go.Figure()
        for row in summary:
            fig.add_trace(
                go.Scatterpolar(
                    r=[row[m] for m in metrics],
                    theta=metrics,
                    fill="toself",
                    name=row["model"],
                )
            )
        fig.update_layout(title="Model Benchmark Radar")
        target = self.output_dir / "benchmark_radar.html"
        fig.write_html(str(target), include_plotlyjs="cdn")
        return target

    def generate_all(self, summary: list[dict[str, Any]]) -> list[Path]:
        """Generate all benchmark plots."""

        return [
            self.latency_chart(summary),
            self.accuracy_chart(summary),
            self.radar_chart(summary),
        ]
