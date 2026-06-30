"""Safe file reader bounded to workspace root."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from crew_platform.tools.base import BaseTool


class FileReaderInput(BaseModel):
    path: str = Field(min_length=1)
    max_chars: int = 10000


class FileReaderOutput(BaseModel):
    content: str


class FileReaderTool(BaseTool[FileReaderInput, FileReaderOutput]):
    name = "file_reader"
    description = "Reads local files from allowed workspace"
    input_model = FileReaderInput
    output_model = FileReaderOutput

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()

    async def run(self, payload: FileReaderInput) -> FileReaderOutput:
        path = (self.workspace_root / payload.path).resolve()
        if self.workspace_root not in path.parents and path != self.workspace_root:
            raise ValueError("Path escapes workspace")
        content = path.read_text(encoding="utf-8")
        return FileReaderOutput(content=content[: payload.max_chars])
