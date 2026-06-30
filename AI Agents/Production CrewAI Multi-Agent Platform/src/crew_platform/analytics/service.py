"""Analytics aggregations and workflow visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from crew_platform.orchestration.models import TaskExecution


class AnalyticsService:
    """Builds analytics summaries and workflow graph artifacts."""

    def summarize(self, tasks: list[TaskExecution]) -> dict[str, Any]:
        if not tasks:
            return {
                "task_completion": 0.0,
                "agent_utilization": {},
                "avg_attempts": 0.0,
                "tool_frequency": {},
                "confidence_avg": 0.0,
            }

        total = len(tasks)
        completed = sum(task.status.value == "completed" for task in tasks)
        agent_utilization: dict[str, int] = {}
        tool_frequency: dict[str, int] = {}
        attempts = 0
        confidences = 0.0

        for task in tasks:
            agent_utilization[task.agent_role] = agent_utilization.get(task.agent_role, 0) + 1
            attempts += max(1, task.attempt)
            confidences += task.confidence
            for tool in task.tools:
                tool_frequency[tool] = tool_frequency.get(tool, 0) + 1

        return {
            "task_completion": completed / total,
            "agent_utilization": agent_utilization,
            "avg_attempts": attempts / total,
            "tool_frequency": tool_frequency,
            "confidence_avg": confidences / total,
        }

    def workflow_figure(self, tasks: list[TaskExecution]) -> go.Figure:
        x_values = list(range(len(tasks)))
        y_values = [1 for _ in tasks]
        labels = [f"{task.task_id}\n{task.agent_role}" for task in tasks]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers+text",
                text=labels,
                textposition="bottom center",
                marker={"size": 16, "color": "#1f77b4"},
            )
        )

        for idx, task in enumerate(tasks):
            for dep in task.dependencies:
                dep_idx = next((i for i, t in enumerate(tasks) if t.task_id == dep), None)
                if dep_idx is None:
                    continue
                fig.add_shape(
                    type="line",
                    x0=dep_idx,
                    y0=1,
                    x1=idx,
                    y1=1,
                    line={"color": "#999", "width": 2},
                )

        fig.update_layout(
            title="Workflow Graph",
            xaxis={"visible": False},
            yaxis={"visible": False},
            showlegend=False,
            height=360,
        )
        return fig

    def save_workflow_html(self, tasks: list[TaskExecution], output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = self.workflow_figure(tasks)
        fig.write_html(str(path), include_plotlyjs="cdn")
        return str(path)
