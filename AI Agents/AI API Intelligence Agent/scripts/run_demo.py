"""Demo runner for quick end-to-end query."""

from __future__ import annotations

import os
from rich import print

from api_intel_agent.agents import AgentRunner


if __name__ == "__main__":
    # Keep demo deterministic in restricted environments.
    os.environ.setdefault("AGENT_DISABLE_CHROMA", "1")
    runner = AgentRunner()
    response = runner.query_sync(
        "python",
        apis=["jsonplaceholder", "openlibrary"],
        use_cache=False,
    )
    print(response.model_dump(mode="json"))
