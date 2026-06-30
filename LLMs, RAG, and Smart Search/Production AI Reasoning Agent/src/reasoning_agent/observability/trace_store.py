"""Trace persistence layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reasoning_agent.utils.json_utils import dumps


class TraceStore:
    """Persist run traces as JSON."""

    def __init__(self, base_dir: str = "logs/traces") -> None:
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def save(self, run_id: str, payload: dict[str, Any]) -> Path:
        """Save trace payload to disk."""

        target = self.base / f"{run_id}.json"
        target.write_text(dumps(payload, indent=True), encoding="utf-8")
        return target
