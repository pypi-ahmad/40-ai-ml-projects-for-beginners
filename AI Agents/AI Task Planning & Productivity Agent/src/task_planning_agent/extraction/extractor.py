"""Task extraction engine combining heuristics and optional LLM refinement."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from task_planning_agent.ingestion.parser import ParsedTaskCandidate, parse_messy_input
from task_planning_agent.schemas import RiskLevel, Task


@dataclass(slots=True)
class ExtractionResult:
    tasks: list[Task]
    dropped_lines: list[str]


class TaskExtractor:
    """Extract structured tasks from messy input."""

    def __init__(self, default_estimate_minutes: int = 45) -> None:
        self.default_estimate_minutes = default_estimate_minutes

    def extract(self, raw_input: str, context: dict[str, Any] | None = None) -> ExtractionResult:
        candidates = parse_messy_input(raw_input)
        tasks: list[Task] = []
        dropped: list[str] = []

        for candidate in candidates:
            task = self._candidate_to_task(candidate, context=context or {})
            if not task.name:
                dropped.append(candidate.raw_line)
                continue
            tasks.append(task)

        return ExtractionResult(tasks=tasks, dropped_lines=dropped)

    def _candidate_to_task(self, candidate: ParsedTaskCandidate, context: dict[str, Any]) -> Task:
        deadline = (
            datetime.fromisoformat(candidate.deadline_text)
            if candidate.deadline_text is not None
            else None
        )
        confidence = 0.55
        if candidate.deadline_text:
            confidence += 0.2
        if candidate.estimated_minutes:
            confidence += 0.15
        if candidate.people:
            confidence += 0.05

        risk_level = RiskLevel.MEDIUM
        if deadline and (deadline - datetime.utcnow()).days <= 1:
            risk_level = RiskLevel.HIGH
        elif deadline is None:
            risk_level = RiskLevel.LOW

        return Task(
            name=candidate.title,
            description=candidate.description,
            deadline=deadline,
            estimated_minutes=candidate.estimated_minutes or self.default_estimate_minutes,
            people=candidate.people,
            project=candidate.project,
            context=context.get("context"),
            confidence=min(0.99, max(0.1, confidence)),
            risk_level=risk_level,
            reasoning="Extracted from messy input using parser heuristics.",
        )
