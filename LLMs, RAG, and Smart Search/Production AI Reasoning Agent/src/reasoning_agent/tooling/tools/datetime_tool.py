"""Datetime tool."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from reasoning_agent.tooling.base import ToolContext, ToolSpec


class DatetimeInput(BaseModel):
    """Datetime input payload."""

    timezone: str = "UTC"


class DatetimeOutput(BaseModel):
    """Datetime output payload."""

    iso: str
    timezone: str
    weekday: str


def current_datetime(payload: DatetimeInput, _: ToolContext) -> DatetimeOutput:
    """Get datetime in requested timezone."""

    tz = ZoneInfo(payload.timezone)
    now = datetime.now(tz=tz)
    return DatetimeOutput(iso=now.isoformat(), timezone=payload.timezone, weekday=now.strftime("%A"))


spec = ToolSpec(
    name="datetime",
    description="Get current date/time in requested timezone",
    input_model=DatetimeInput,
    output_model=DatetimeOutput,
    tags=["time", "utility"],
)
