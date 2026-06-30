"""Core schemas for planning, scheduling, and analytics."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class PriorityStrategy(str, Enum):
    EISENHOWER = "eisenhower"
    MOSCOW = "moscow"
    ABCDE = "abcde"
    RICE = "rice"
    ICE = "ice"
    WSJF = "wsjf"
    URGENCY_IMPORTANCE = "urgency_importance"
    WEIGHTED = "weighted"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskStatus(str, Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class Task(BaseModel):
    """Structured task representation."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    deadline: datetime | None = None
    estimated_minutes: int = 30
    dependencies: list[str] = Field(default_factory=list)
    is_meeting: bool = False
    reminder_at: datetime | None = None
    recurring_rule: str | None = None
    priority_hint: str | None = None
    project: str | None = None
    context: str | None = None
    people: list[str] = Field(default_factory=list)
    location: str | None = None
    required_tools: list[str] = Field(default_factory=list)
    confidence: float = 0.6
    source_ref: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    priority_score: float = 0.0
    reasoning: str = ""
    risk_level: RiskLevel = RiskLevel.MEDIUM


class TaskDependency(BaseModel):
    parent_task_id: str
    child_task_id: str
    reason: str = ""


class ScheduleBlock(BaseModel):
    task_id: str
    task_name: str
    suggested_start_time: datetime
    suggested_end_time: datetime
    priority: float
    confidence: float
    reasoning: str
    risk_level: RiskLevel


class DailyPlan(BaseModel):
    date: str
    blocks: list[ScheduleBlock] = Field(default_factory=list)
    breaks: list[ScheduleBlock] = Field(default_factory=list)


class WeeklyPlan(BaseModel):
    week_of: str
    daily_plans: list[DailyPlan] = Field(default_factory=list)


class SprintPlan(BaseModel):
    sprint_name: str
    start_date: str
    end_date: str
    blocks: list[ScheduleBlock] = Field(default_factory=list)


class UserPreference(BaseModel):
    user_id: str
    working_hours_start: str = "09:00"
    working_hours_end: str = "18:00"
    focus_hours: list[str] = Field(default_factory=lambda: ["09:00-12:00"])
    preferred_break_minutes: int = 15
    meeting_preference: str = "afternoon"
    energy_curve: dict[str, int] = Field(
        default_factory=lambda: {"morning": 90, "afternoon": 70, "evening": 40}
    )


class ReflectionRecord(BaseModel):
    plan_id: str
    what_worked: list[str] = Field(default_factory=list)
    what_failed: list[str] = Field(default_factory=list)
    missed_deadlines: list[str] = Field(default_factory=list)
    overruns: list[str] = Field(default_factory=list)
    context_switching_issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Recommendation(BaseModel):
    category: str
    suggestion: str
    impact: str


class AnalyticsSnapshot(BaseModel):
    completed_tasks: int = 0
    completion_rate: float = 0.0
    average_delay_minutes: float = 0.0
    planning_accuracy: float = 0.0
    focus_time_minutes: int = 0
    deep_work_minutes: int = 0
    meetings_minutes: int = 0
    context_switches: int = 0
    energy_score: float = 0.0
    burnout_score: float = 0.0
    weekly_productivity_score: float = 0.0


class ScheduleResponse(BaseModel):
    task: str
    priority: float
    deadline: datetime | None
    estimated_duration: int
    dependencies: list[str]
    suggested_start_time: datetime
    suggested_end_time: datetime
    confidence: float
    reasoning: str
    risk_level: RiskLevel


class PlanRequest(BaseModel):
    user_id: str
    raw_input: str
    strategy: PriorityStrategy = PriorityStrategy.WSJF
    timezone: str = "Asia/Kolkata"


class ReplanRequest(BaseModel):
    user_id: str
    reason: str
    additional_input: str = ""


class PlanReport(BaseModel):
    plan_id: str
    user_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    summary: str
    schedule: list[ScheduleResponse] = Field(default_factory=list)
    reflections: list[str] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    analytics: AnalyticsSnapshot = Field(default_factory=AnalyticsSnapshot)


class PlanSession(BaseModel):
    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    raw_input: str
    tasks: list[Task] = Field(default_factory=list)
    dependencies: list[TaskDependency] = Field(default_factory=list)
    schedule_blocks: list[ScheduleBlock] = Field(default_factory=list)
    reflection: ReflectionRecord | None = None
    recommendations: list[Recommendation] = Field(default_factory=list)
    analytics: AnalyticsSnapshot = Field(default_factory=AnalyticsSnapshot)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    output: Any
    error: str | None = None


class ConnectorStatus(BaseModel):
    name: str
    enabled: bool
    healthy: bool
    capabilities: list[str] = Field(default_factory=list)
    message: str = ""
