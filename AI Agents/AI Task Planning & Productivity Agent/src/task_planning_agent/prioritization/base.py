"""Prioritization strategy interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from task_planning_agent.schemas import Task


class PrioritizationStrategy(ABC):
    """Interface for strategy-specific task scoring."""

    @abstractmethod
    def score(self, task: Task) -> tuple[float, str]:
        """Return normalized score [0, 100] and rationale."""
