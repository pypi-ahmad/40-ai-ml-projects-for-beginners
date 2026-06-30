"""Prioritization strategy registry and batch scoring."""

from __future__ import annotations

from task_planning_agent.prioritization.base import PrioritizationStrategy
from task_planning_agent.prioritization.strategies import (
    AbcdeStrategy,
    EisenhowerStrategy,
    IceStrategy,
    MoscowStrategy,
    RiceStrategy,
    UrgencyImportanceStrategy,
    WeightedStrategy,
    WsjfStrategy,
)
from task_planning_agent.schemas import PriorityStrategy, Task


STRATEGY_REGISTRY: dict[PriorityStrategy, PrioritizationStrategy] = {
    PriorityStrategy.EISENHOWER: EisenhowerStrategy(),
    PriorityStrategy.MOSCOW: MoscowStrategy(),
    PriorityStrategy.ABCDE: AbcdeStrategy(),
    PriorityStrategy.RICE: RiceStrategy(),
    PriorityStrategy.ICE: IceStrategy(),
    PriorityStrategy.WSJF: WsjfStrategy(),
    PriorityStrategy.URGENCY_IMPORTANCE: UrgencyImportanceStrategy(),
    PriorityStrategy.WEIGHTED: WeightedStrategy(),
}


def score_tasks(tasks: list[Task], strategy: PriorityStrategy) -> list[Task]:
    scorer = STRATEGY_REGISTRY[strategy]
    for task in tasks:
        score, rationale = scorer.score(task)
        task.priority_score = round(score, 2)
        task.reasoning = f"{task.reasoning} | Priority: {rationale}".strip()
    return sorted(tasks, key=lambda t: t.priority_score, reverse=True)
