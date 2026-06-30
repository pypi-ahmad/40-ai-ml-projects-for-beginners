"""LLM-as-a-judge scoring for responses."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from reasoning_agent.evals.dataset import BenchmarkPrompt
from reasoning_agent.llm.ollama import OllamaProvider
from reasoning_agent.utils.output_parser import parse_structured_output


class JudgeOutput(BaseModel):
    correctness: float
    grounding: float
    completeness: float
    tool_usage: float
    notes: str


@dataclass(slots=True)
class LLMJudge:
    """Evaluates answer quality with local model."""

    llm: OllamaProvider
    model: str

    async def score(self, prompt: BenchmarkPrompt, answer: str, used_tools: list[str]) -> JudgeOutput:
        judge_prompt = (
            "You are strict evaluator. Return JSON only with keys: "
            "correctness, grounding, completeness, tool_usage, notes. "
            "Score each numeric field from 0 to 1.\n\n"
            f"Prompt: {prompt.prompt}\n"
            f"Expected keywords: {prompt.expected_keywords}\n"
            f"Required tools: {prompt.required_tools}\n"
            f"Used tools: {used_tools}\n"
            f"Answer: {answer}\n"
        )
        try:
            response = await self.llm.generate(
                prompt=judge_prompt,
                model=self.model,
                temperature=0.0,
                max_tokens=350,
                raw=True,
            )
            return parse_structured_output(response.text, JudgeOutput, retries=2)
        except Exception:  # noqa: BLE001
            return JudgeOutput(
                correctness=0.0,
                grounding=0.0,
                completeness=0.0,
                tool_usage=0.0,
                notes="judge unavailable",
            )
