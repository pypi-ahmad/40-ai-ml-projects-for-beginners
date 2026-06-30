"""Analytics aggregation and MLflow logging."""

from __future__ import annotations

import os
from typing import Any

import mlflow

from task_planning_agent.analytics.duckdb_store import DuckDBStore
from task_planning_agent.schemas import AnalyticsSnapshot, ScheduleBlock, Task


class AnalyticsEngine:
    """Compute and track productivity metrics."""

    def __init__(self, duckdb_path: str, mlflow_tracking_uri: str) -> None:
        self.store = DuckDBStore(duckdb_path)
        if mlflow_tracking_uri.startswith("file:"):
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    def snapshot(
        self,
        user_id: str,
        plan_id: str,
        tasks: list[Task],
        blocks: list[ScheduleBlock],
    ) -> AnalyticsSnapshot:
        completed = sum(1 for task in tasks if task.status.value == "completed")
        completion_rate = (completed / len(tasks)) if tasks else 0.0
        meetings_minutes = int(
            sum(
                (block.suggested_end_time - block.suggested_start_time).total_seconds() / 60
                for block in blocks
                if "meeting" in block.task_name.lower()
            )
        )
        focus_minutes = int(
            sum(
                (block.suggested_end_time - block.suggested_start_time).total_seconds() / 60
                for block in blocks
                if block.priority >= 70.0
            )
        )
        context_switches = max(0, len(blocks) - 1)
        deep_work_minutes = int(
            sum(
                (block.suggested_end_time - block.suggested_start_time).total_seconds() / 60
                for block in blocks
                if (block.suggested_end_time - block.suggested_start_time).total_seconds() >= 75 * 60
            )
        )
        delay = float(sum(1 for block in blocks if block.risk_level.value == "high") * 15)
        planning_accuracy = max(0.0, 100.0 - delay)

        energy_score = max(0.0, 100.0 - context_switches * 2.5)
        burnout_score = min(100.0, (meetings_minutes / 60) * 10 + max(0, context_switches - 5) * 5)
        weekly_productivity_score = max(
            0.0,
            min(100.0, completion_rate * 100 + deep_work_minutes / 6 - burnout_score * 0.2),
        )

        snapshot = AnalyticsSnapshot(
            completed_tasks=completed,
            completion_rate=completion_rate,
            average_delay_minutes=delay,
            planning_accuracy=planning_accuracy,
            focus_time_minutes=focus_minutes,
            deep_work_minutes=deep_work_minutes,
            meetings_minutes=meetings_minutes,
            context_switches=context_switches,
            energy_score=energy_score,
            burnout_score=burnout_score,
            weekly_productivity_score=weekly_productivity_score,
        )

        self._log(user_id=user_id, plan_id=plan_id, snapshot=snapshot)
        return snapshot

    def _log(self, user_id: str, plan_id: str, snapshot: AnalyticsSnapshot) -> None:
        metrics: dict[str, Any] = snapshot.model_dump()
        try:
            with mlflow.start_run(run_name=f"plan-{plan_id}", nested=False):
                for metric, value in metrics.items():
                    if isinstance(value, (float, int)):
                        mlflow.log_metric(metric, float(value))
                        self.store.log_metric(
                            user_id=user_id,
                            plan_id=plan_id,
                            metric=metric,
                            value=float(value),
                        )
        except Exception:
            for metric, value in metrics.items():
                if isinstance(value, (float, int)):
                    self.store.log_metric(user_id=user_id, plan_id=plan_id, metric=metric, value=float(value))
