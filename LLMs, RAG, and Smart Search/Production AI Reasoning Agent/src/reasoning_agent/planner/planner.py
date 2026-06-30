"""Task decomposition planner."""

from __future__ import annotations

from reasoning_agent.llm.ollama_client import OllamaClient
from reasoning_agent.parsing.output_parser import StructuredOutputParser
from reasoning_agent.prompts import planning_prompt
from reasoning_agent.schemas import PlanningOutput


class Planner:
    """Generate multi-step plans from user input."""

    def __init__(self, llm: OllamaClient, model: str, temperature: float, max_tokens: int) -> None:
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.parser = StructuredOutputParser(llm, model)

    def plan(self, user_input: str, tools: list[dict[str, object]]) -> PlanningOutput:
        """Plan with model-first, heuristic fallback."""

        prompt = planning_prompt(user_input, tools)
        try:
            result = self.llm.generate(
                self.model,
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return self.parser.parse(result.text, PlanningOutput)
        except Exception:
            return PlanningOutput(
                objective=user_input,
                steps=[
                    "Understand request and required data",
                    "Collect evidence using most relevant tool",
                    "Synthesize concise grounded answer",
                ],
                reasoning_summary="Fallback planner used due to model/parsing failure.",
                required_tools=[],
            )
