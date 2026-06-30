"""JSON explorer tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from reasoning_agent.tools.base import BaseTool


class JSONExplorerInput(BaseModel):
    path: str


class JSONExplorerOutput(BaseModel):
    top_level_keys: list[str]
    item_count: int


class JSONExplorerTool(BaseTool[JSONExplorerInput, JSONExplorerOutput]):
    name = "json_explorer"
    description = "Inspect JSON files quickly"
    input_model = JSONExplorerInput
    output_model = JSONExplorerOutput

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()

    async def run(self, payload: JSONExplorerInput) -> JSONExplorerOutput:
        path = (self.workspace_root / payload.path).resolve()
        if self.workspace_root not in path.parents and path != self.workspace_root:
            raise ValueError("Path escapes workspace")
        data: Any = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return JSONExplorerOutput(top_level_keys=list(data.keys()), item_count=len(data))
        if isinstance(data, list):
            return JSONExplorerOutput(top_level_keys=[], item_count=len(data))
        return JSONExplorerOutput(top_level_keys=[], item_count=1)
