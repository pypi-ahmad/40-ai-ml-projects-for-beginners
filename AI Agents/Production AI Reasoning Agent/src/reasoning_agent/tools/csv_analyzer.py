"""CSV analyzer tool."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from pydantic import BaseModel

from reasoning_agent.tools.base import BaseTool


class CSVAnalyzerInput(BaseModel):
    path: str


class CSVAnalyzerOutput(BaseModel):
    rows: int
    columns: int
    column_names: list[str]
    null_counts: dict[str, int]


class CSVAnalyzerTool(BaseTool[CSVAnalyzerInput, CSVAnalyzerOutput]):
    name = "csv_analyzer"
    description = "Returns schema and summary statistics for CSV"
    input_model = CSVAnalyzerInput
    output_model = CSVAnalyzerOutput

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()

    async def run(self, payload: CSVAnalyzerInput) -> CSVAnalyzerOutput:
        path = (self.workspace_root / payload.path).resolve()
        if self.workspace_root not in path.parents and path != self.workspace_root:
            raise ValueError("Path escapes workspace")
        df = pl.read_csv(path)
        null_counts = {col: int(df[col].null_count()) for col in df.columns}
        return CSVAnalyzerOutput(
            rows=df.height,
            columns=df.width,
            column_names=df.columns,
            null_counts=null_counts,
        )
