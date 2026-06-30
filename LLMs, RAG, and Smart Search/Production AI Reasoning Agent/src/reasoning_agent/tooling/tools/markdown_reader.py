"""Markdown reader tool."""

from __future__ import annotations

from pydantic import BaseModel

from reasoning_agent.tooling.base import ToolContext, ToolSpec


class MarkdownReaderInput(BaseModel):
    """Markdown reader input payload."""

    path: str


class MarkdownReaderOutput(BaseModel):
    """Markdown reader output payload."""

    path: str
    headings: list[str]
    preview: str


def read_markdown(payload: MarkdownReaderInput, context: ToolContext) -> MarkdownReaderOutput:
    """Read markdown and extract heading list."""

    target = (context.workspace_root / payload.path).resolve()
    root = context.workspace_root.resolve()
    if root not in target.parents and target != root:
        raise ValueError("Path outside workspace root is not allowed")

    text = target.read_text(encoding="utf-8", errors="replace")
    headings = [line.strip("# ").strip() for line in text.splitlines() if line.startswith("#")]
    return MarkdownReaderOutput(
        path=str(target.relative_to(root)),
        headings=headings,
        preview=text[:3000],
    )


spec = ToolSpec(
    name="markdown_reader",
    description="Read markdown file, extract headings and preview",
    input_model=MarkdownReaderInput,
    output_model=MarkdownReaderOutput,
    tags=["filesystem", "docs"],
)
