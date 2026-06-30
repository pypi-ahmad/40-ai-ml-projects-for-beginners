from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from reasoning_agent.tooling import ToolContext, ToolRegistry, ToolSpec


class InModel(BaseModel):
    x: int


class OutModel(BaseModel):
    y: int


def test_tool_registry_register_and_invoke(tmp_path: Path) -> None:
    registry = ToolRegistry(workspace_root=tmp_path)

    def handler(payload: InModel, _: ToolContext) -> OutModel:
        return OutModel(y=payload.x + 1)

    registry.register(
        ToolSpec(
            name="inc",
            description="increment",
            input_model=InModel,
            output_model=OutModel,
        ),
        handler,
    )

    result = registry.invoke(
        "inc",
        {"x": 4},
        ToolContext(session_id="s", run_id="r", workspace_root=tmp_path),
    )
    assert result.ok is True
    assert result.output == {"y": 5}


def test_tool_registry_validation_error(tmp_path: Path) -> None:
    registry = ToolRegistry(workspace_root=tmp_path)

    def handler(payload: InModel, _: ToolContext) -> OutModel:
        return OutModel(y=payload.x + 1)

    registry.register(
        ToolSpec(
            name="inc",
            description="increment",
            input_model=InModel,
            output_model=OutModel,
        ),
        handler,
    )

    result = registry.invoke(
        "inc",
        {"x": "bad"},
        ToolContext(session_id="s", run_id="r", workspace_root=tmp_path),
    )
    assert result.ok is False
    assert result.validation_passed is False
