"""Dataset schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Sample:
    task_type: str
    instruction: str = ""
    input: str = ""
    output: str = ""
    label: str | int | float | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetStats:
    size: int
    avg_input_chars: float
    avg_output_chars: float
    unique_ratio: float
