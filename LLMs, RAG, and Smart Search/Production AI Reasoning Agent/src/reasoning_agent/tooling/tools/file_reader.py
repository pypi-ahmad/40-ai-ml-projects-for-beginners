"""Workspace-restricted file reader."""

from __future__ import annotations

from pydantic import BaseModel, Field

from reasoning_agent.tooling.base import ToolContext, ToolSpec


class FileReaderInput(BaseModel):
    """File reader input payload."""

    path: str = Field(description="Path relative to workspace root")
    max_chars: int = Field(default=6000, ge=100, le=20000)


class FileReaderOutput(BaseModel):
    """File reader output payload."""

    path: str
    content: str


def read_file(payload: FileReaderInput, context: ToolContext) -> FileReaderOutput:
    """Read local file within workspace boundary."""

    target = (context.workspace_root / payload.path).resolve()
    root = context.workspace_root.resolve()
    if root not in target.parents and target != root:
        raise ValueError("Path outside workspace root is not allowed")
    if not target.exists() or not target.is_file():
        raise ValueError("File not found")

    content = target.read_text(encoding="utf-8", errors="replace")
    return FileReaderOutput(path=str(target.relative_to(root)), content=content[: payload.max_chars])


spec = ToolSpec(
    name="file_reader",
    description="Read UTF-8 text files within workspace boundary",
    input_model=FileReaderInput,
    output_model=FileReaderOutput,
    tags=["filesystem", "local"],
)
