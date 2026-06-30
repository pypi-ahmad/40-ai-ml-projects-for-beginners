"""JSON explorer tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from reasoning_agent.tooling.base import ToolContext, ToolSpec
from reasoning_agent.utils.json_utils import loads


class JSONExplorerInput(BaseModel):
    """JSON explorer input payload."""

    path: str
    key_path: str = ""


class JSONExplorerOutput(BaseModel):
    """JSON explorer output payload."""

    path: str
    key_path: str
    value: Any


def _resolve_key_path(data: Any, key_path: str) -> Any:
    if not key_path:
        return data
    current = data
    for token in key_path.split("."):
        if isinstance(current, dict):
            current = current[token]
        elif isinstance(current, list):
            current = current[int(token)]
        else:
            raise ValueError(f"Cannot descend into type: {type(current).__name__}")
    return current


def explore_json(payload: JSONExplorerInput, context: ToolContext) -> JSONExplorerOutput:
    """Load JSON file and return nested key path value."""

    target = (context.workspace_root / payload.path).resolve()
    root = context.workspace_root.resolve()
    if root not in target.parents and target != root:
        raise ValueError("Path outside workspace root is not allowed")

    content = target.read_text(encoding="utf-8")
    data = loads(content)
    value = _resolve_key_path(data, payload.key_path)
    return JSONExplorerOutput(path=str(target.relative_to(root)), key_path=payload.key_path, value=value)


spec = ToolSpec(
    name="json_explorer",
    description="Navigate JSON via dot key path and array indexes",
    input_model=JSONExplorerInput,
    output_model=JSONExplorerOutput,
    tags=["data", "filesystem"],
)
