"""Local file and structured data tools."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from internet_agent.tools.base import BaseTool


class PathInput(BaseModel):
    path: str


class TextOutput(BaseModel):
    content: str


class JsonOutput(BaseModel):
    data: dict[str, Any] | list[Any]


class RowsOutput(BaseModel):
    rows: list[dict[str, Any]]


class FileReaderTool(BaseTool[PathInput, TextOutput]):
    name = "file_reader"
    description = "Read plain text files from local filesystem."
    input_model = PathInput
    output_model = TextOutput

    async def run(self, payload: PathInput) -> TextOutput:
        text = Path(payload.path).read_text(encoding="utf-8")
        return TextOutput(content=text)


class CSVReaderTool(BaseTool[PathInput, RowsOutput]):
    name = "csv_reader"
    description = "Read CSV into list of rows."
    input_model = PathInput
    output_model = RowsOutput

    async def run(self, payload: PathInput) -> RowsOutput:
        with Path(payload.path).open("r", encoding="utf-8") as file_handle:
            rows = list(csv.DictReader(file_handle))
        return RowsOutput(rows=rows)


class JSONReaderTool(BaseTool[PathInput, JsonOutput]):
    name = "json_reader"
    description = "Read JSON file from local filesystem."
    input_model = PathInput
    output_model = JsonOutput

    async def run(self, payload: PathInput) -> JsonOutput:
        data = json.loads(Path(payload.path).read_text(encoding="utf-8"))
        return JsonOutput(data=data)
