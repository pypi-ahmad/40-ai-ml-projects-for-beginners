"""Hybrid judge scoring (deterministic + LLM rubric)."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from reasoning_agent.llm.ollama_client import OllamaClient
from reasoning_agent.parsing.output_parser import StructuredOutputParser


@dataclass(slots=True)
class JudgeInput:
    """Judge input tuple."""

    prompt: str
    answer: str
    expected_keywords: list[str]


class JudgeOutput(BaseModel):
    """Judge structured score."""

    reasoning_quality: float = Field(ge=0.0, le=1.0)
    correctness: float = Field(ge=0.0, le=1.0)
    grounding: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    tool_usage: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class JudgeRunner:
    """Hybrid judge with deterministic baseline plus model rubric."""

    def __init__(self, llm: OllamaClient, judge_model: str = "granite4.1:3b") -> None:
        self.llm = llm
        self.judge_model = judge_model
        self.parser = StructuredOutputParser(llm, judge_model, retries=1)

    def score(self, item: JudgeInput) -> dict[str, float | str]:
        """Return hybrid score record."""

        det = self._deterministic_score(item.answer, item.expected_keywords)
        rubric = self._llm_rubric(item)

        final_score = 0.6 * det + 0.4 * rubric["overall"]
        return {
            "deterministic": det,
            "judge_overall": rubric["overall"],
            "hybrid": final_score,
            "judge_rationale": rubric["rationale"],
        }

    def _deterministic_score(self, answer: str, expected_keywords: list[str]) -> float:
        if not expected_keywords:
            return 0.5
        lower = answer.lower()
        hit = sum(1 for kw in expected_keywords if kw in lower)
        return float(hit) / float(len(expected_keywords))

    def _llm_rubric(self, item: JudgeInput) -> dict[str, float | str]:
        prompt = (
            "You are evaluation judge. Return strict JSON with keys: "
            "reasoning_quality, correctness, grounding, completeness, tool_usage, rationale. "
            "Each score in [0,1].\n"
            f"Prompt: {item.prompt}\n"
            f"Expected keywords: {item.expected_keywords}\n"
            f"Answer: {item.answer}"
        )

        try:
            res = self.llm.generate(self.judge_model, prompt, temperature=0.0, max_tokens=400)
            parsed = self.parser.parse(res.text, JudgeOutput)
            overall = (
                parsed.reasoning_quality
                + parsed.correctness
                + parsed.grounding
                + parsed.completeness
                + parsed.tool_usage
            ) / 5.0
            return {"overall": overall, "rationale": parsed.rationale}
        except Exception:
            return {"overall": 0.5, "rationale": "judge fallback"}
