"""Benchmark dataset definitions and loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import orjson
from pydantic import BaseModel, Field


class BenchmarkPrompt(BaseModel):
    """Single benchmark item."""

    prompt_id: str
    category: str
    prompt: str
    expected_keywords: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    current_events: bool = False


class BenchmarkPrediction(BaseModel):
    """Prediction row emitted by benchmark runner."""

    prompt_id: str
    category: str
    model: str
    answer: str
    success: bool
    latency_ms: float
    iterations: int
    tool_calls: int
    retry_count: int
    required_tools: list[str] = Field(default_factory=list)
    used_tools: list[str] = Field(default_factory=list)
    keyword_score: float = 0.0
    tool_selection_score: float = 0.0
    judge_score: float | None = None
    judge_notes: str | None = None


def load_benchmark_prompts(path: Path) -> list[BenchmarkPrompt]:
    """Load benchmark prompts from JSONL file."""

    items: list[BenchmarkPrompt] = []
    with path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            data = orjson.loads(line)
            if "id" in data:
                data["prompt_id"] = data.pop("id")
            items.append(BenchmarkPrompt.model_validate(data))
    return items


def iter_prompt_chunks(prompts: list[BenchmarkPrompt], chunk_size: int = 25) -> Iterator[list[BenchmarkPrompt]]:
    """Yield prompt chunks for batched benchmark execution."""

    for start in range(0, len(prompts), chunk_size):
        yield prompts[start : start + chunk_size]
