"""Benchmark dataset loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reasoning_agent.utils.json_utils import loads


@dataclass(slots=True)
class EvalPrompt:
    """Single evaluation prompt record."""

    prompt_id: str
    category: str
    prompt: str
    expected_keywords: list[str]


class BenchmarkDataset:
    """Dataset container and validators."""

    def __init__(self, records: list[EvalPrompt]) -> None:
        self.records = records

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "BenchmarkDataset":
        """Load benchmark prompts from JSONL file."""

        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Benchmark dataset not found: {target}")

        rows: list[EvalPrompt] = []
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = loads(line)
            rows.append(
                EvalPrompt(
                    prompt_id=str(data["id"]),
                    category=str(data["category"]),
                    prompt=str(data["prompt"]),
                    expected_keywords=[str(x).lower() for x in data.get("expected_keywords", [])],
                )
            )

        if len(rows) < 100:
            raise ValueError("Benchmark dataset must contain at least 100 prompts")

        return cls(rows)
