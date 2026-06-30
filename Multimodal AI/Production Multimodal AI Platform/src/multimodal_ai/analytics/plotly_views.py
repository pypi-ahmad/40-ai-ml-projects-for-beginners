"""Plotly chart builders for analytics."""

from __future__ import annotations

from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go


def latency_bar(summary: dict[str, dict[str, float]]) -> go.Figure:
    """Build latency bar chart."""

    labels = list(summary.keys())
    values = [summary[key].get("avg_ms", 0.0) for key in labels]
    return px.bar(x=labels, y=values, labels={"x": "Capability", "y": "Avg Latency (ms)"})


def model_usage_radar(rows: list[dict[str, float | int | str]]) -> go.Figure:
    """Build model usage radar chart."""

    if not rows:
        return go.Figure()

    metrics = [row["capability"] for row in rows]
    values = [float(row["calls"]) for row in rows]
    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values,
                theta=metrics,
                fill="toself",
                name="Model Usage",
            )
        ]
    )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
    return fig


def save_figure(fig: go.Figure, output_path: Path) -> None:
    """Persist figure to html."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
