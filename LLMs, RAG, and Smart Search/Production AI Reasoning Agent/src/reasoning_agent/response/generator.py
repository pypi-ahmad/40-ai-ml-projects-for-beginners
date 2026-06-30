"""Final response synthesis."""

from __future__ import annotations

from reasoning_agent.llm.ollama_client import OllamaClient
from reasoning_agent.parsing.output_parser import StructuredOutputParser
from reasoning_agent.prompts import final_answer_prompt
from reasoning_agent.schemas import FinalAnswerOutput


class ResponseGenerator:
    """Generate final grounded answer."""

    def __init__(self, llm: OllamaClient, model: str, temperature: float, max_tokens: int) -> None:
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.parser = StructuredOutputParser(llm, model)

    def generate(
        self,
        *,
        user_input: str,
        plan_steps: list[str],
        observations: list[dict[str, object]],
        trace_summary: list[str],
    ) -> FinalAnswerOutput:
        """Generate answer with model-first fallback."""

        prompt = final_answer_prompt(user_input, plan_steps, observations, trace_summary)
        try:
            result = self.llm.generate(
                self.model,
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return self.parser.parse(result.text, FinalAnswerOutput)
        except Exception:
            obs_lines = [f"- {obs}" for obs in observations[-5:]]
            answer = "\n".join(["Model unavailable. Fallback response using tool observations:", *obs_lines])
            return FinalAnswerOutput(answer=answer, citations=[], completeness_score=0.2)
