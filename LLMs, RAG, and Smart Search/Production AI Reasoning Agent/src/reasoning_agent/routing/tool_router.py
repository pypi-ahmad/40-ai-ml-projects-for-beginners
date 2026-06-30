"""Tool routing logic."""

from __future__ import annotations

import re
from typing import Any

from reasoning_agent.llm.ollama_client import OllamaClient
from reasoning_agent.parsing.output_parser import StructuredOutputParser
from reasoning_agent.prompts import tool_routing_prompt
from reasoning_agent.schemas import ToolRoutingOutput


class ToolRouter:
    """Select best next tool based on plan and observations."""

    def __init__(self, llm: OllamaClient, model: str, temperature: float, max_tokens: int) -> None:
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.parser = StructuredOutputParser(llm, model)

    def route(
        self,
        *,
        user_input: str,
        step: str,
        tools: list[dict[str, Any]],
        observations: list[dict[str, Any]],
    ) -> ToolRoutingOutput:
        """Route with model-first, heuristic fallback."""

        prompt = tool_routing_prompt(user_input, step, tools, observations)
        try:
            result = self.llm.generate(
                self.model,
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            routed = self.parser.parse(result.text, ToolRoutingOutput)
            return routed
        except Exception:
            return self._heuristic_route(user_input, step)

    def _heuristic_route(self, user_input: str, step: str) -> ToolRoutingOutput:
        text = f"{user_input} {step}".lower()

        if any(key in text for key in ["weather", "temperature", "forecast"]):
            return ToolRoutingOutput(
                tool_name="weather",
                arguments={"location": user_input},
                justification="weather intent detected",
            )

        if any(key in text for key in ["currency", "usd", "eur", "exchange"]):
            return ToolRoutingOutput(
                tool_name="currency_converter",
                arguments={"amount": 1.0, "from_currency": "USD", "to_currency": "EUR"},
                justification="currency intent detected",
            )

        math_pattern = re.compile(r"([0-9][0-9\s\+\-\*\/\(\)\.\%]+)")
        if any(key in text for key in ["calculate", "math", "mean", "median"]) or math_pattern.search(text):
            matched = math_pattern.search(user_input)
            expression = matched.group(1).strip() if matched else user_input
            return ToolRoutingOutput(
                tool_name="calculator",
                arguments={"expression": expression},
                justification="math expression detected",
            )

        if any(key in text for key in ["python", "code", "script"]):
            return ToolRoutingOutput(
                tool_name="python_repl",
                arguments={"code": user_input},
                justification="python execution intent detected",
            )

        if any(key in text for key in ["wiki", "wikipedia", "who is", "what is"]):
            return ToolRoutingOutput(
                tool_name="wikipedia",
                arguments={"query": user_input},
                justification="encyclopedic lookup intent detected",
            )

        if any(key in text for key in ["search", "latest", "news", "find", "current"]):
            return ToolRoutingOutput(
                tool_name="duckduckgo_search",
                arguments={"query": user_input, "max_results": 5},
                justification="search intent detected",
            )

        return ToolRoutingOutput(
            tool_name="response_generator",
            arguments={},
            justification="No external tool required",
        )
