"""Markdown reader tool."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from crew_platform.tools.base import BaseTool


class MarkdownReaderInput(BaseModel):
    path: str


class MarkdownReaderOutput(BaseModel):
    content: str


class MarkdownReaderTool(BaseTool[MarkdownReaderInput, MarkdownReaderOutput]):
    name = "markdown_reader"
    description = "Read markdown files"
    input_model = MarkdownReaderInput
    output_model = MarkdownReaderOutput

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()

    async def run(self, payload: MarkdownReaderInput) -> MarkdownReaderOutput:
        path = (self.workspace_root / payload.path).resolve()
        if self.workspace_root not in path.parents and path != self.workspace_root:
            raise ValueError("Path escapes workspace")
        return MarkdownReaderOutput(content=path.read_text(encoding="utf-8"))
