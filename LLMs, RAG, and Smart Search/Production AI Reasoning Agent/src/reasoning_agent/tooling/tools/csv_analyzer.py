"""CSV analyzer tool."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

from reasoning_agent.tooling.base import ToolContext, ToolSpec


class CSVAnalyzerInput(BaseModel):
    """CSV analyzer input payload."""

    path: str
    sample_rows: int = 5


class CSVAnalyzerOutput(BaseModel):
    """CSV analyzer output payload."""

    path: str
    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]
    sample: list[dict[str, object]]


def analyze_csv(payload: CSVAnalyzerInput, context: ToolContext) -> CSVAnalyzerOutput:
    """Read CSV and return quick profile summary."""

    target = (context.workspace_root / payload.path).resolve()
    root = context.workspace_root.resolve()
    if root not in target.parents and target != root:
        raise ValueError("Path outside workspace root is not allowed")
    if not target.exists():
        raise ValueError("CSV file not found")

    frame = pd.read_csv(target)
    return CSVAnalyzerOutput(
        path=str(target.relative_to(root)),
        rows=int(frame.shape[0]),
        columns=int(frame.shape[1]),
        column_names=[str(c) for c in frame.columns.tolist()],
        dtypes={str(k): str(v) for k, v in frame.dtypes.to_dict().items()},
        sample=frame.head(payload.sample_rows).to_dict(orient="records"),
    )


spec = ToolSpec(
    name="csv_analyzer",
    description="Profile CSV schema, dtypes, and sample rows",
    input_model=CSVAnalyzerInput,
    output_model=CSVAnalyzerOutput,
    tags=["data", "filesystem"],
)
