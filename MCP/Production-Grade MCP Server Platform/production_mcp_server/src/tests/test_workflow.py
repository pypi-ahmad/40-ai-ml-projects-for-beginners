from __future__ import annotations

import pytest

from server.platform import Platform


@pytest.mark.asyncio
async def test_workflow_executes() -> None:
    platform = Platform.from_config("configs/default.yaml")
    result = await platform.workflow_engine.run("Compute 2+2 and create report")
    assert result["status"] in {"completed", "degraded"}
    assert "steps" in result
