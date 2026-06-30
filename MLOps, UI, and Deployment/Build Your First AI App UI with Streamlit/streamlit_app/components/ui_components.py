"""Reusable UI rendering components for Streamlit pages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.utils.helpers import FIGURES_DIR, format_latency


def _save_plotly_figure(fig: go.Figure, stem: str) -> Path:
    """Persist figure as HTML and best-effort PNG for portfolio artifacts."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    html_path = FIGURES_DIR / f"{stem}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")

    png_path = FIGURES_DIR / f"{stem}.png"
    try:
        fig.write_image(str(png_path))
    except Exception:
        # PNG export requires kaleido and local chrome; HTML is always kept.
        pass
    return html_path


def render_metric_card(label: str, value: str, delta: str | None = None) -> None:
    st.metric(label=label, value=value, delta=delta)


def render_result_box(content: str, title: str = "Result") -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.write(content)


def render_model_comparison(results: list[dict[str, Any]]) -> go.Figure:
    """Render summary comparison table + chart from benchmark rows."""
    if not results:
        st.warning("No benchmark results to display yet.")
        return go.Figure()

    df = pd.DataFrame(results)
    for column in ["mean_latency", "mean_throughput_wps", "mean_memory_mb", "mean_quality_score"]:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    st.dataframe(df, use_container_width=True)

    fig = px.bar(
        df,
        x="model",
        y="mean_latency",
        color="model",
        title="Mean Latency by Model (seconds)",
        labels={"model": "Model", "mean_latency": "Mean latency (s)"},
    )
    st.plotly_chart(fig, use_container_width=True)
    _save_plotly_figure(fig, "benchmark_mean_latency")
    return fig


def render_latency_distribution(run_rows: list[dict[str, Any]]) -> go.Figure:
    """Render box plot for run-level latency distribution."""
    if not run_rows:
        st.warning("No run-level benchmark rows available.")
        return go.Figure()

    df = pd.DataFrame(run_rows)
    fig = px.box(
        df,
        x="model",
        y="latency_seconds",
        color="model",
        points="all",
        title="Latency Distribution per Model",
        labels={"latency_seconds": "Latency (seconds)", "model": "Model"},
    )
    st.plotly_chart(fig, use_container_width=True)
    _save_plotly_figure(fig, "benchmark_latency_distribution")
    return fig


def render_throughput_chart(results: list[dict[str, Any]]) -> go.Figure:
    """Render throughput chart and persist artifact."""
    if not results:
        return go.Figure()

    df = pd.DataFrame(results)
    fig = px.bar(
        df,
        x="model",
        y="mean_throughput_wps",
        color="model",
        title="Mean Throughput by Model (words/second)",
        labels={"mean_throughput_wps": "Words/second", "model": "Model"},
    )
    st.plotly_chart(fig, use_container_width=True)
    _save_plotly_figure(fig, "benchmark_throughput")
    return fig


def render_quality_radar(results: list[dict[str, Any]]) -> go.Figure:
    """Render radar chart for latency/throughput/memory/quality tradeoff."""
    if not results:
        return go.Figure()

    df = pd.DataFrame(results).copy()
    latency_max = max(df["mean_latency"].max(), 1e-9)
    memory_max = max(df["mean_memory_mb"].max(), 1e-9)
    throughput_max = max(df["mean_throughput_wps"].max(), 1e-9)

    # Normalize metrics to 0-1 so multiple units can share a radar chart.
    df["latency_score"] = 1 - (df["mean_latency"] / latency_max)
    df["memory_score"] = 1 - (df["mean_memory_mb"] / memory_max)
    df["throughput_score"] = df["mean_throughput_wps"] / throughput_max

    categories = ["latency_score", "throughput_score", "memory_score", "mean_quality_score"]
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=[row[metric] for metric in categories],
                theta=["Latency", "Throughput", "Memory", "Quality"],
                fill="toself",
                name=row["model"],
            )
        )

    fig.update_layout(
        title="Model Tradeoff Radar (higher is better)",
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)
    _save_plotly_figure(fig, "benchmark_tradeoff_radar")
    return fig


def render_usage_stats(total_requests: int, avg_latency: float, total_tokens: int, active_model: str) -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Requests", str(total_requests))
    with col2:
        render_metric_card("Avg Latency", format_latency(avg_latency))
    with col3:
        render_metric_card("Output Tokens (proxy)", f"{total_tokens:,}")
    with col4:
        render_metric_card("Active Model", active_model)


def render_confidence_gauge(confidence: float, label: str = "Confidence") -> go.Figure:
    """Render confidence gauge used by classification and sentiment pages."""
    value = max(0.0, min(1.0, confidence)) * 100
    color = "#16a34a" if value >= 70 else "#d97706" if value >= 40 else "#dc2626"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%"},
            title={"text": label},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "#fee2e2"},
                    {"range": [40, 70], "color": "#fef3c7"},
                    {"range": [70, 100], "color": "#dcfce7"},
                ],
            },
        )
    )
    fig.update_layout(height=260)
    st.plotly_chart(fig, use_container_width=True)
    return fig


def render_cache_comparison_chart(cache_metrics: dict[str, float]) -> go.Figure:
    """Visualize latency difference between uncached and cached requests."""
    df = pd.DataFrame(
        {
            "mode": ["uncached", "cached_first", "cached_second"],
            "seconds": [
                cache_metrics.get("uncached_seconds", 0.0),
                cache_metrics.get("cached_first_seconds", 0.0),
                cache_metrics.get("cached_second_seconds", 0.0),
            ],
        }
    )
    fig = px.bar(df, x="mode", y="seconds", title="Cache Impact on Inference Time")
    st.plotly_chart(fig, use_container_width=True)
    _save_plotly_figure(fig, "cache_impact")
    return fig
