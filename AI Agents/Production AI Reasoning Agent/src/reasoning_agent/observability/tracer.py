"""JSONL trace writer."""

from __future__ import annotations

from pathlib import Path

import orjson
from loguru import logger

from reasoning_agent.observability.events import EventRecord


class JsonlTracer:
    """Append-only JSONL tracer for run events."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: EventRecord) -> None:
        data = event.model_dump(mode="json")
        with self.output_path.open("ab") as handle:
            handle.write(orjson.dumps(data) + b"\n")

    def safe_log(self, event: EventRecord) -> None:
        try:
            self.log(event)
        except Exception as exc:  # noqa: BLE001
            logger.warning("trace_write_failed: {}", exc)
