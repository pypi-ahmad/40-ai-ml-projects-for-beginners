from __future__ import annotations

import os

os.environ["CREW_PLATFORM_DISABLE_CHROMA"] = "1"

from crew_platform.api.main import agents, health


def test_health_endpoint_function() -> None:
    payload = health()
    assert payload["status"] == "ok"


def test_agents_endpoint_function() -> None:
    payload = agents()
    assert len(payload.get("agents", [])) >= 15
