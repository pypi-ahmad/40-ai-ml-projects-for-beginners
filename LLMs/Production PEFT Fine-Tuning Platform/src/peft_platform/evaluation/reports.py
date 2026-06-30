"""Evaluation report writers."""

from __future__ import annotations

from pathlib import Path

from peft_platform.utils.io import ensure_dir


def write_markdown_report(path: Path, title: str, lines: list[str]) -> Path:
    ensure_dir(path.parent)
    content = [f"# {title}", ""] + [f"- {line}" for line in lines]
    path.write_text("\n".join(content), encoding="utf-8")
    return path
