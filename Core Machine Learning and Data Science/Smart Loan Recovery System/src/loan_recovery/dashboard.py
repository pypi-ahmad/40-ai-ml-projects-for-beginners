"""Plotly dashboard components for portfolio and recovery intelligence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class DashboardBuilder:
    """Build executive-friendly interactive plots."""

    @staticmethod
    def portfolio_health_kpis(scored_df: pd.DataFrame) -> dict[str, float]:
        """Return top-level portfolio KPI dictionary."""
        return {
            "borrowers": float(len(scored_df)),
            "total_outstanding": float(scored_df["Outstanding_Balance"].sum()),
            "avg_recovery_probability": float(scored_df["recovery_probability"].mean()),
            "avg_risk_score": float(scored_df["risk_score"].mean()),
            "high_risk_share": float((scored_df["risk_tier"].isin(["High", "Very High"])).mean()),
        }

    @staticmethod
    def risk_distribution(scored_df: pd.DataFrame) -> go.Figure:
        """Histogram of risk scores by risk tier."""
        fig = px.histogram(
            scored_df,
            x="risk_score",
            color="risk_tier",
            nbins=30,
            barmode="overlay",
            title="Risk Score Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(xaxis_title="Risk Score", yaxis_title="Borrower Count")
        return fig

    @staticmethod
    def recovery_probability_distribution(scored_df: pd.DataFrame) -> go.Figure:
        """Histogram of recovery probabilities."""
        fig = px.histogram(
            scored_df,
            x="recovery_probability",
            color="risk_tier",
            nbins=30,
            title="Recovery Probability Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(xaxis_title="Recovery Probability", yaxis_title="Borrower Count")
        return fig

    @staticmethod
    def segment_distribution(assigned_df: pd.DataFrame, segment_col: str = "segment") -> go.Figure:
        """Borrower counts per segment with risk-tier coloring."""
        if "segment_name" in assigned_df.columns:
            segment_col = "segment_name"
        if segment_col not in assigned_df.columns:
            return go.Figure()

        seg = (
            assigned_df.groupby([segment_col, "risk_tier"], as_index=False)
            .agg(borrowers=("Borrower_ID", "count"), avg_risk=("risk_score", "mean"))
            .sort_values("avg_risk", ascending=False)
        )

        fig = px.bar(
            seg,
            x=segment_col,
            y="borrowers",
            color="risk_tier",
            title="Segment Distribution by Risk Tier",
            barmode="stack",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(xaxis_title="Segment", yaxis_title="Borrowers")
        return fig

    @staticmethod
    def strategy_recommendations(assigned_df: pd.DataFrame) -> go.Figure:
        """Recommended strategy counts and expected recovery value by strategy."""
        strategy_df = (
            assigned_df.groupby("recommended_strategy", as_index=False)
            .agg(
                borrowers=("Borrower_ID", "count"),
                expected_recovery_value=("expected_recovery_value", "sum"),
            )
            .sort_values("expected_recovery_value", ascending=False)
        )

        fig = px.bar(
            strategy_df,
            y="recommended_strategy",
            x="borrowers",
            color="expected_recovery_value",
            orientation="h",
            title="Recovery Strategy Prioritization",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(yaxis_title="Strategy", xaxis_title="Borrowers")
        return fig

    @staticmethod
    def scenario_comparison(scenario_df: pd.DataFrame) -> go.Figure:
        """Compare risk and expected recovery across what-if scenarios."""
        fig = px.bar(
            scenario_df,
            x="scenario",
            y="expected_recovery_value",
            color="avg_risk_score",
            title="What-If Scenario Comparison",
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(xaxis_title="Scenario", yaxis_title="Expected Recovery Value")
        return fig

    @staticmethod
    def save_html(fig: go.Figure, path: Path) -> Path:
        """Persist interactive figure as standalone HTML."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs="cdn")
        return path
