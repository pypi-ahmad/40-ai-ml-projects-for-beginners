"""Datetime tool."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel

from reasoning_agent.tools.base import BaseTool


class DatetimeInput(BaseModel):
    timezone_name: str = "UTC"


class DatetimeOutput(BaseModel):
    iso_datetime: str


class DatetimeTool(BaseTool[DatetimeInput, DatetimeOutput]):
    name = "datetime"
    description = "Returns current datetime in UTC"
    input_model = DatetimeInput
    output_model = DatetimeOutput

    async def run(self, payload: DatetimeInput) -> DatetimeOutput:
        _ = payload
        return DatetimeOutput(iso_datetime=datetime.now(timezone.utc).isoformat())
