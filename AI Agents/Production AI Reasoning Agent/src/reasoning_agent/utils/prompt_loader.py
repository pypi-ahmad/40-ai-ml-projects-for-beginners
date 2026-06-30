"""Prompt template loader."""

from __future__ import annotations

from pathlib import Path

from reasoning_agent.constants import PROMPTS_DIR


def load_prompt(name: str) -> str:
    """Load prompt template by filename stem."""

    path = Path(PROMPTS_DIR) / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")
