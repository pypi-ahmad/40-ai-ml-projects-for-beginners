"""Consensus reasoning with triad agents."""

from __future__ import annotations

from dataclasses import dataclass

from crew_platform.config import Settings
from crew_platform.llm.ollama import OllamaProvider


@dataclass(slots=True)
class ConsensusResult:
    selected_answer: str
    candidate_answers: list[str]
    rationale: str


class ConsensusService:
    """Runs three independent model passes, then selects best answer."""

    def __init__(self, settings: Settings, llm: OllamaProvider) -> None:
        self.settings = settings
        self.llm = llm

    async def run(self, objective: str, context: str) -> ConsensusResult:
        answers: list[str] = []
        for model in self.settings.llm.consensus_models[:3]:
            prompt = (
                "Solve objective independently. Return concise evidence-backed answer.\n"
                f"Objective: {objective}\n"
                f"Context:\n{context}\n"
            )
            try:
                out = await self.llm.generate(
                    prompt=prompt,
                    model=model,
                    temperature=0.2,
                    max_tokens=min(800, self.settings.llm.max_tokens),
                )
                answers.append(out.text)
            except Exception:
                answers.append(f"Model {model} unavailable")

        scored = sorted(answers, key=self._score_answer, reverse=True)
        best = scored[0] if scored else ""
        return ConsensusResult(
            selected_answer=best,
            candidate_answers=answers,
            rationale="Selected answer has strongest evidence density and completeness.",
        )

    @staticmethod
    def _score_answer(answer: str) -> float:
        score = min(len(answer) / 1200.0, 1.0)
        if "http" in answer:
            score += 0.2
        if "risk" in answer.lower() or "limitation" in answer.lower():
            score += 0.1
        return score
