"""Structured output parser with retries and fallback."""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


class ParseError(RuntimeError):
    """Raised when structured parsing fails after retries."""


def parse_structured_output(raw_text: str, model: type[ModelT], retries: int = 2) -> ModelT:
    """Parse JSON model output with repair retries.

    Args:
        raw_text: Raw model output.
        model: Pydantic model.
        retries: Number of normalization retries.

    Returns:
        Parsed model instance.
    """

    candidate = raw_text.strip()
    for _ in range(retries + 1):
        try:
            data = json.loads(candidate)
            return model.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            candidate = _extract_json_block(candidate)
    raise ParseError(f"Could not parse model output into {model.__name__}")


def _extract_json_block(text: str) -> str:
    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1 and right > left:
        return text[left : right + 1]
    return text
