"""Structured output parser with retries and fallback."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ValidationError

from reasoning_agent.llm.ollama_client import OllamaClient
from reasoning_agent.prompts import parser_repair_prompt
from reasoning_agent.utils.json_utils import loads

T = TypeVar("T", bound=BaseModel)


class StructuredOutputParser:
    """Parse model output into validated schema with repair retries."""

    def __init__(self, llm: OllamaClient, model: str, retries: int = 2) -> None:
        self.llm = llm
        self.model = model
        self.retries = retries

    def parse(self, raw_output: str, schema: type[T]) -> T:
        """Parse raw model output into schema, retrying with repair prompts."""

        attempts = 0
        candidate = raw_output
        last_error: Exception | None = None

        while attempts <= self.retries:
            try:
                data = loads(candidate)
                return schema.model_validate(data)
            except (ValidationError, ValueError, TypeError) as exc:
                last_error = exc
                if attempts == self.retries:
                    break
                repair = parser_repair_prompt(candidate, self.model, schema.__name__)
                repaired = self.llm.generate(self.model, repair)
                candidate = repaired.text
                attempts += 1
                continue

        # Fallback: attempt to locate JSON object boundaries.
        try:
            start = raw_output.find("{")
            end = raw_output.rfind("}")
            if start >= 0 and end > start:
                data = loads(raw_output[start : end + 1])
                return schema.model_validate(data)
        except Exception:
            pass

        raise ValueError(f"Failed to parse {schema.__name__}: {last_error}")
