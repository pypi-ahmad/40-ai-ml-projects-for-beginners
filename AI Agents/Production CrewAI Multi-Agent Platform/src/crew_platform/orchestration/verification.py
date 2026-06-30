"""Verification, QA, reflection, and confidence scoring."""

from __future__ import annotations

import re

from crew_platform.config import Settings
from crew_platform.llm.ollama import OllamaProvider
from crew_platform.orchestration.models import TaskExecution, VerificationResult


class VerificationService:
    """Applies fact-checking/QA/reflection scoring to task outputs."""

    def __init__(self, settings: Settings, llm: OllamaProvider) -> None:
        self.settings = settings
        self.llm = llm

    async def verify(self, tasks: list[TaskExecution]) -> VerificationResult:
        if not tasks:
            return VerificationResult(issues=["No tasks executed"], needs_rerun=True)

        factual_score = self._factual_score(tasks)
        qa_score = self._qa_score(tasks)
        reflection_score = await self._reflection_score(tasks)
        confidence = round((factual_score + qa_score + reflection_score) / 3.0, 3)

        issues: list[str] = []
        if factual_score < 0.5:
            issues.append("Low factual evidence coverage")
        if qa_score < 0.5:
            issues.append("Low task completion quality")
        if reflection_score < 0.5:
            issues.append("Reflection indicates weak answer confidence")

        threshold = self.settings.orchestration.consensus_trigger_confidence
        return VerificationResult(
            factual_score=factual_score,
            qa_score=qa_score,
            reflection_score=reflection_score,
            confidence=confidence,
            issues=issues,
            needs_rerun=confidence < threshold,
        )

    def _factual_score(self, tasks: list[TaskExecution]) -> float:
        with_refs = 0
        for task in tasks:
            text = self._task_text(task)
            if "http" in text or "source" in text.lower() or "citation" in text.lower():
                with_refs += 1
        return round(with_refs / max(len(tasks), 1), 3)

    def _qa_score(self, tasks: list[TaskExecution]) -> float:
        completed = sum(task.status.value == "completed" for task in tasks)
        no_errors = sum(task.error in (None, "") for task in tasks)
        ratio = (completed / max(len(tasks), 1)) * 0.7 + (no_errors / max(len(tasks), 1)) * 0.3
        return round(ratio, 3)

    async def _reflection_score(self, tasks: list[TaskExecution]) -> float:
        summary = "\n".join(self._task_text(t)[:300] for t in tasks)
        prompt = (
            "Score confidence 0..1 for these multi-agent outputs. "
            "Return plain number only.\n"
            f"{summary}"
        )
        try:
            res = await self.llm.generate(
                prompt=prompt,
                model=self.settings.llm.reflection_model,
                temperature=0.0,
                max_tokens=20,
                raw=True,
            )
            match = re.search(r"0(?:\.\d+)?|1(?:\.0+)?", res.text)
            if match:
                return round(float(match.group(0)), 3)
        except Exception:
            pass

        heuristic = sum(len(self._task_text(t)) > 80 for t in tasks) / max(len(tasks), 1)
        return round(min(1.0, 0.4 + 0.6 * heuristic), 3)

    @staticmethod
    def _task_text(task: TaskExecution) -> str:
        if not task.result:
            return ""
        return str(task.result.get("content", task.result))
