"""Self-reflection and plan revision."""

from __future__ import annotations

from reasoning_agent.llm.ollama_client import OllamaClient
from reasoning_agent.parsing.output_parser import StructuredOutputParser
from reasoning_agent.prompts import reflection_prompt
from reasoning_agent.schemas import ReflectionOutput


class Reflector:
    """Evaluate progress and revise plan when needed."""

    def __init__(self, llm: OllamaClient, model: str, temperature: float, max_tokens: int) -> None:
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.parser = StructuredOutputParser(llm, model)

    def reflect(
        self,
        *,
        user_input: str,
        plan_steps: list[str],
        observations: list[dict[str, object]],
        errors: list[str],
    ) -> ReflectionOutput:
        """Run reflection model call with fallback."""

        prompt = reflection_prompt(user_input, plan_steps, observations, errors)
        try:
            result = self.llm.generate(
                self.model,
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return self.parser.parse(result.text, ReflectionOutput)
        except Exception:
            success = bool(observations and all(obs.get("ok", False) for obs in observations[-1:]))
            return ReflectionOutput(success=success, confidence=0.4 if success else 0.1, revised_plan=[], notes="fallback reflection")
