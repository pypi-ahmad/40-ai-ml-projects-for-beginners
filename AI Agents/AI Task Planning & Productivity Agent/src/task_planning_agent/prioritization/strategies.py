"""Prioritization strategy implementations."""

from __future__ import annotations

from datetime import datetime

from task_planning_agent.prioritization.base import PrioritizationStrategy
from task_planning_agent.schemas import Task


def _deadline_urgency(task: Task) -> float:
    if task.deadline is None:
        return 30.0
    now = datetime.now(tz=task.deadline.tzinfo) if task.deadline.tzinfo else datetime.now()
    hours = max(1.0, (task.deadline - now).total_seconds() / 3600)
    return min(100.0, 120.0 / hours * 10)


def _effort_penalty(task: Task) -> float:
    return min(40.0, task.estimated_minutes / 12)


class EisenhowerStrategy(PrioritizationStrategy):
    def score(self, task: Task) -> tuple[float, str]:
        urgency = _deadline_urgency(task)
        importance = 80.0 if "critical" in task.description.lower() else 55.0
        score = min(100.0, urgency * 0.5 + importance * 0.5)
        return score, "Eisenhower urgency/importance blend"


class MoscowStrategy(PrioritizationStrategy):
    def score(self, task: Task) -> tuple[float, str]:
        text = task.description.lower()
        if "must" in text:
            return 95.0, "MoSCoW: Must"
        if "should" in text:
            return 75.0, "MoSCoW: Should"
        if "could" in text:
            return 55.0, "MoSCoW: Could"
        return 35.0, "MoSCoW: Won't/unknown"


class AbcdeStrategy(PrioritizationStrategy):
    def score(self, task: Task) -> tuple[float, str]:
        text = task.priority_hint.lower() if task.priority_hint else ""
        mapping = {"a": 95.0, "b": 80.0, "c": 65.0, "d": 45.0, "e": 25.0}
        return mapping.get(text[:1], 60.0), "ABCDE mapping"


class RiceStrategy(PrioritizationStrategy):
    def score(self, task: Task) -> tuple[float, str]:
        reach = 50.0 + len(task.people) * 5
        impact = 70.0 if task.project else 50.0
        confidence = task.confidence * 100
        effort = max(1.0, task.estimated_minutes / 30)
        score = min(100.0, (reach * impact * confidence) / (10000 * effort) * 100)
        return score, "RICE normalized score"


class IceStrategy(PrioritizationStrategy):
    def score(self, task: Task) -> tuple[float, str]:
        impact = 70.0 if "revenue" in task.description.lower() else 55.0
        confidence = task.confidence * 100
        ease = max(5.0, 100 - task.estimated_minutes / 3)
        score = min(100.0, (impact + confidence + ease) / 3)
        return score, "ICE average"


class WsjfStrategy(PrioritizationStrategy):
    def score(self, task: Task) -> tuple[float, str]:
        user_business = 60.0 if task.project else 50.0
        time_criticality = _deadline_urgency(task)
        risk_reduction = 80.0 if task.risk_level.value == "high" else 45.0
        job_size = max(5.0, task.estimated_minutes / 6)
        score = min(100.0, (user_business + time_criticality + risk_reduction) / job_size * 4)
        return score, "WSJF weighted shortest job first"


class UrgencyImportanceStrategy(PrioritizationStrategy):
    def score(self, task: Task) -> tuple[float, str]:
        urgency = _deadline_urgency(task)
        importance = 75.0 if task.project else 50.0
        score = urgency * 0.6 + importance * 0.4
        return min(100.0, score), "Urgency vs Importance"


class WeightedStrategy(PrioritizationStrategy):
    def score(self, task: Task) -> tuple[float, str]:
        urgency = _deadline_urgency(task)
        importance = 80.0 if "critical" in task.description.lower() else 50.0
        impact = 70.0 if task.people else 45.0
        confidence = task.confidence * 100
        effort_penalty = _effort_penalty(task)
        score = urgency * 0.3 + importance * 0.35 + impact * 0.2 + confidence * 0.1 - effort_penalty
        return max(0.0, min(100.0, score)), "Custom weighted strategy"
