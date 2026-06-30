"""Visualization recommendation and rendering helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ai_sql_assistant.types import VisualizationSpec


def _column_kind(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def recommend_visualizations(rows: list[dict[str, Any]]) -> list[VisualizationSpec]:
    """Recommend visualization options based on result shape."""
    if not rows:
        return [VisualizationSpec(chart_type="table", reason="No rows returned.")]

    frame = pd.DataFrame(rows)
    numeric = [col for col in frame.columns if _column_kind(frame[col]) == "numeric"]
    categorical = [col for col in frame.columns if _column_kind(frame[col]) == "categorical"]
    temporal_keywords = ("date", "month", "year", "time", "day", "week", "quarter")
    temporal = [
        col
        for col in frame.columns
        if any(key in col.lower() for key in temporal_keywords) or _column_kind(frame[col]) == "datetime"
    ]

    specs = [VisualizationSpec(chart_type="table", reason="Always provide raw table view.")]

    if categorical and numeric:
        specs.append(
            VisualizationSpec(
                chart_type="bar",
                x=categorical[0],
                y=numeric[0],
                reason="Categorical split with numeric metric.",
            )
        )

    if temporal and numeric:
        specs.append(
            VisualizationSpec(
                chart_type="time_series",
                x=temporal[0],
                y=numeric[0],
                reason="Temporal trend with numeric value.",
            )
        )
        specs.append(
            VisualizationSpec(
                chart_type="line",
                x=temporal[0],
                y=numeric[0],
                reason="Line view for trend analysis.",
            )
        )

    if len(numeric) >= 2:
        specs.append(
            VisualizationSpec(
                chart_type="scatter",
                x=numeric[0],
                y=numeric[1],
                reason="Numeric relationship analysis.",
            )
        )

    if numeric:
        specs.append(
            VisualizationSpec(
                chart_type="histogram",
                x=numeric[0],
                reason="Distribution of numeric metric.",
            )
        )

    if len(categorical) >= 1 and len(numeric) >= 1:
        specs.append(
            VisualizationSpec(
                chart_type="pie",
                x=categorical[0],
                y=numeric[0],
                reason="Share breakdown across categories.",
            )
        )

    if len(numeric) >= 2:
        specs.append(
            VisualizationSpec(
                chart_type="heatmap",
                x=numeric[0],
                y=numeric[1],
                reason="Density view for paired metrics.",
            )
        )

    # Deduplicate by chart type while preserving order.
    unique: dict[str, VisualizationSpec] = {}
    for spec in specs:
        unique.setdefault(spec.chart_type, spec)

    return list(unique.values())


def render_chart(rows: list[dict[str, Any]], spec: VisualizationSpec) -> go.Figure | None:
    """Render Plotly figure from rows and selected visualization spec."""
    if not rows or spec.chart_type == "table":
        return None

    frame = pd.DataFrame(rows)
    if spec.x and "date" in spec.x.lower() and spec.x in frame.columns:
        frame[spec.x] = pd.to_datetime(frame[spec.x], errors="ignore")

    if spec.chart_type in {"bar", "line", "time_series", "pie", "histogram", "scatter"}:
        if spec.x and spec.x not in frame.columns:
            return None
        if spec.y and spec.y not in frame.columns:
            return None

    if spec.chart_type == "bar":
        return px.bar(frame, x=spec.x, y=spec.y, title="Bar Chart")
    if spec.chart_type in {"line", "time_series"}:
        return px.line(frame, x=spec.x, y=spec.y, title="Time Series")
    if spec.chart_type == "pie":
        return px.pie(frame, names=spec.x, values=spec.y, title="Pie Chart")
    if spec.chart_type == "scatter":
        return px.scatter(frame, x=spec.x, y=spec.y, title="Scatter Plot")
    if spec.chart_type == "histogram":
        return px.histogram(frame, x=spec.x, title="Histogram")
    if spec.chart_type == "heatmap":
        if not spec.x or not spec.y:
            return None
        return px.density_heatmap(frame, x=spec.x, y=spec.y, title="Heatmap")

    return None
