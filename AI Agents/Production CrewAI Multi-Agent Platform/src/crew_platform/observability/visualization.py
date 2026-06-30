"""Plotly visualizations for traces and benchmarks."""

from __future__ import annotations

import polars as pl
import plotly.express as px
import plotly.graph_objects as go


def latency_bar(df: pl.DataFrame):
    return px.bar(df.to_pandas(), x="model", y="avg_latency_ms", title="Average Latency by Model")


def success_rate_bar(df: pl.DataFrame):
    return px.bar(df.to_pandas(), x="model", y="success_rate", title="Success Rate by Model")


def tool_usage_chart(df: pl.DataFrame):
    return px.box(df.to_pandas(), x="model", y="avg_tool_calls", title="Tool Usage Distribution")


def radar_quality(df: pl.DataFrame):
    fig = go.Figure()
    metrics = ["avg_keyword_score", "avg_tool_selection_score", "avg_judge_score"]
    for row in df.to_dicts():
        values = [float(row.get(metric, 0.0)) for metric in metrics]
        fig.add_trace(
            go.Scatterpolar(r=values + [values[0]], theta=metrics + [metrics[0]], fill="toself", name=row["model"])
        )
    fig.update_layout(title="Reasoning Quality Radar")
    return fig
