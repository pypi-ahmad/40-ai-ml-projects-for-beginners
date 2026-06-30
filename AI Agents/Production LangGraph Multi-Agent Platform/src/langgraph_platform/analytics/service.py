"""Analytics service over persisted runs and graph traces."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from langgraph_platform.memory.sqlite_store import SQLiteStore


class AnalyticsService:
    """Compute analytics metrics and charts."""

    def __init__(self, sqlite_store: SQLiteStore) -> None:
        self.sqlite_store = sqlite_store

    def summary(self) -> dict[str, Any]:
        """Return aggregate workflow analytics."""

        runs = self.sqlite_store.list_recent_runs(limit=500)
        if not runs:
            return {
                "workflow_count": 0,
                "avg_confidence": 0.0,
                "verification_pass_rate": 0.0,
            }

        frame = pd.DataFrame(runs)
        pass_rate = (frame["verification_status"] == "passed").mean()
        return {
            "workflow_count": len(frame),
            "avg_confidence": float(frame["confidence"].mean()),
            "verification_pass_rate": float(pass_rate),
        }

    def confidence_trend(self) -> go.Figure:
        """Build confidence-over-time chart."""

        runs = self.sqlite_store.list_recent_runs(limit=100)
        frame = pd.DataFrame(runs)
        if frame.empty:
            return go.Figure()
        frame["created_at"] = pd.to_datetime(frame["created_at"])
        return px.line(
            frame.sort_values("created_at"),
            x="created_at",
            y="confidence",
            title="Confidence Trend",
        )

    def status_distribution(self) -> go.Figure:
        """Build verification status distribution chart."""

        runs = self.sqlite_store.list_recent_runs(limit=200)
        frame = pd.DataFrame(runs)
        if frame.empty:
            return go.Figure()
        counts = frame["verification_status"].value_counts().reset_index()
        counts.columns = ["status", "count"]
        return px.bar(counts, x="status", y="count", title="Verification Status Distribution")
