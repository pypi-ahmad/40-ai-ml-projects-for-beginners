"""Visualization engine with Plotly charts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class VisualizationEngine:
    """Generate chart objects and exports."""

    def bar(self, df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        return px.bar(df, x=x, y=y, title=title)

    def line(self, df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        return px.line(df, x=x, y=y, title=title)

    def scatter(self, df: pd.DataFrame, x: str, y: str, color: str | None = None, title: str = "") -> go.Figure:
        return px.scatter(df, x=x, y=y, color=color, title=title)

    def pie(self, df: pd.DataFrame, names: str, values: str, title: str) -> go.Figure:
        return px.pie(df, names=names, values=values, title=title)

    def histogram(self, df: pd.DataFrame, column: str, title: str) -> go.Figure:
        return px.histogram(df, x=column, title=title)

    def heatmap(self, df: pd.DataFrame, title: str = "Correlation Heatmap") -> go.Figure:
        corr = df.select_dtypes(include=["number"]).corr(numeric_only=True)
        return px.imshow(corr, text_auto=True, title=title)

    def box(self, df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        return px.box(df, x=x, y=y, title=title)

    def violin(self, df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        return px.violin(df, x=x, y=y, box=True, points="all", title=title)

    def correlation_matrix(self, df: pd.DataFrame) -> go.Figure:
        return self.heatmap(df, "Correlation Matrix")

    def time_series_dashboard(self, df: pd.DataFrame, date_col: str, value_col: str, title: str) -> go.Figure:
        work = df.copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col])
        work = work.sort_values(date_col)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=work[date_col], y=work[value_col], mode="lines", name=value_col))
        rolling = pd.to_numeric(work[value_col], errors="coerce").rolling(window=7, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=work[date_col], y=rolling, mode="lines", name="7-day avg"))
        fig.update_layout(title=title, xaxis_title=date_col, yaxis_title=value_col)
        return fig

    def save_html(self, figure: go.Figure, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(str(output_path), include_plotlyjs="cdn")
        return output_path
