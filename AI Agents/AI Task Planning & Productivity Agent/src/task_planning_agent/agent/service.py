"""Application service for planning and re-planning flows."""

from __future__ import annotations

from task_planning_agent.agent.graph import build_graph
from task_planning_agent.agent.nodes import WorkflowNodes
from task_planning_agent.analytics.metrics import AnalyticsEngine
from task_planning_agent.config import AppConfig
from task_planning_agent.memory.manager import MemoryManager
from task_planning_agent.scheduling.optimizer import ScheduleOptimizer
from task_planning_agent.schemas import PlanReport, PriorityStrategy


class PlanningService:
    """High-level façade for API/CLI/UI usage."""

    def __init__(self, config: AppConfig) -> None:
        paths = config.paths
        memory_cfg = config.memory
        analytics_cfg = config.raw.get("analytics", {})
        scheduling_cfg = config.scheduling

        self.memory = MemoryManager(
            sqlite_path=paths.get("sqlite_path", "data/processed/productivity_agent.db"),
            chroma_dir=paths.get("chroma_dir", "data/processed/chroma"),
            collection_name=memory_cfg.get("chroma_collection", "plan_memory"),
        )
        self.analytics = AnalyticsEngine(
            duckdb_path="data/processed/analytics.duckdb",
            mlflow_tracking_uri=analytics_cfg.get("mlflow_tracking_uri", "file:./artifacts/mlruns"),
        )
        self.scheduler = ScheduleOptimizer(
            workday_start=scheduling_cfg.get("workday_start", "09:00"),
            workday_end=scheduling_cfg.get("workday_end", "18:00"),
            break_minutes=scheduling_cfg.get("default_break_minutes", 15),
            deep_work_block_minutes=scheduling_cfg.get("deep_work_block_minutes", 90),
        )

        self.nodes = WorkflowNodes(memory=self.memory, analytics=self.analytics, scheduler=self.scheduler)
        self.graph = build_graph(self.nodes)

    def plan(
        self,
        *,
        user_id: str,
        raw_input: str,
        strategy: PriorityStrategy,
        timezone: str,
    ) -> PlanReport:
        output = self.graph.invoke(
            {
                "user_id": user_id,
                "raw_input": raw_input,
                "strategy": strategy,
                "timezone": timezone,
            }
        )
        return output["report"]

    def replan(self, *, user_id: str, reason: str, additional_input: str = "") -> PlanReport:
        history = self.memory.history(user_id=user_id, limit=1)
        if not history:
            return self.plan(
                user_id=user_id,
                raw_input=f"Replan request: {reason}\n{additional_input}",
                strategy=PriorityStrategy.WSJF,
                timezone="Asia/Kolkata",
            )

        previous = history[0]
        merged_input = (
            f"Previous input:\n{previous.raw_input}\n\n"
            f"Replan reason: {reason}\n"
            f"Additional input:\n{additional_input}"
        )
        return self.plan(
            user_id=user_id,
            raw_input=merged_input,
            strategy=PriorityStrategy.WSJF,
            timezone="Asia/Kolkata",
        )
