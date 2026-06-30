"""Typed state for LangGraph planning workflow."""

from __future__ import annotations

from typing import TypedDict

from task_planning_agent.schemas import (
    AnalyticsSnapshot,
    PlanReport,
    PriorityStrategy,
    Recommendation,
    ReflectionRecord,
    ScheduleBlock,
    Task,
    TaskDependency,
)


class AgentState(TypedDict, total=False):
    user_id: str
    raw_input: str
    strategy: PriorityStrategy
    timezone: str
    tasks: list[Task]
    dependencies: list[TaskDependency]
    schedule_blocks: list[ScheduleBlock]
    issues: list[str]
    reflection: ReflectionRecord
    recommendations: list[Recommendation]
    analytics: AnalyticsSnapshot
    report: PlanReport
    plan_id: str
