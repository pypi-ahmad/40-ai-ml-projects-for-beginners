"""Analytics computations and chart generation."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import plotly.express as px

from api_intel_agent.core.schemas import ChartArtifact, ConnectorResult


class AnalyticsEngine:
    def __init__(self, output_dir: str = "artifacts/charts") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def repository_rankings(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked = sorted(records, key=lambda x: x.get("stargazers_count", 0), reverse=True)
        return ranked[:20]

    def language_distribution(self, records: list[dict[str, Any]]) -> dict[str, int]:
        return dict(Counter((item.get("language") or "Unknown") for item in records))

    def api_latency_summary(self, connector_results: list[ConnectorResult]) -> list[dict[str, Any]]:
        return [
            {
                "provider": result.provider,
                "latency_ms": result.latency_ms or 0,
                "status": result.status,
            }
            for result in connector_results
        ]

    def trend_timeline(self, records: list[dict[str, Any]], date_field: str) -> list[dict[str, Any]]:
        timeline: dict[str, int] = {}
        for item in records:
            raw = item.get(date_field)
            if not raw:
                continue
            date_key = str(raw)[:10]
            timeline[date_key] = timeline.get(date_key, 0) + 1
        return [{"date": key, "count": value} for key, value in sorted(timeline.items())]

    def generate_charts(
        self,
        repo_rankings: list[dict[str, Any]],
        language_dist: dict[str, int],
        latency_summary: list[dict[str, Any]],
    ) -> list[ChartArtifact]:
        artifacts: list[ChartArtifact] = []
        now = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

        if repo_rankings:
            fig = px.bar(
                repo_rankings[:10],
                x="name",
                y="stargazers_count",
                title="Repository Star Rankings",
            )
            out = self.output_dir / f"repo_rankings_{now}.html"
            fig.write_html(out)
            artifacts.append(ChartArtifact(title="Repository rankings", kind="bar", path=str(out)))

        if language_dist:
            data = [{"language": lang, "count": count} for lang, count in language_dist.items()]
            fig = px.pie(data, values="count", names="language", title="Language Distribution")
            out = self.output_dir / f"language_distribution_{now}.html"
            fig.write_html(out)
            artifacts.append(ChartArtifact(title="Language distribution", kind="pie", path=str(out)))

        if latency_summary:
            fig = px.line(latency_summary, x="provider", y="latency_ms", title="API Latency by Provider")
            out = self.output_dir / f"api_latency_{now}.html"
            fig.write_html(out)
            artifacts.append(ChartArtifact(title="API latency", kind="line", path=str(out)))

        if repo_rankings:
            timeline_rows = self.trend_timeline(repo_rankings, date_field="created_at")
            if timeline_rows:
                fig = px.area(timeline_rows, x="date", y="count", title="Repository Trend Timeline")
                out = self.output_dir / f"trend_timeline_{now}.html"
                fig.write_html(out)
                artifacts.append(ChartArtifact(title="Trend timeline", kind="area", path=str(out)))

            heatmap_rows = [
                {
                    "language": row.get("language") or "Unknown",
                    "repo": row.get("name") or "repo",
                    "stars": row.get("stargazers_count", 0),
                }
                for row in repo_rankings[:30]
            ]
            if heatmap_rows:
                fig = px.density_heatmap(
                    heatmap_rows,
                    x="language",
                    y="repo",
                    z="stars",
                    title="Repository Heatmap",
                )
                out = self.output_dir / f"repo_heatmap_{now}.html"
                fig.write_html(out)
                artifacts.append(ChartArtifact(title="Repository heatmap", kind="heatmap", path=str(out)))

        return artifacts
