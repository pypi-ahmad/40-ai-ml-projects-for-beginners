"""Weather tool (keyless, fallback messaging)."""

from __future__ import annotations

from pydantic import BaseModel

from crew_platform.tools.base import BaseTool


class WeatherInput(BaseModel):
    location: str


class WeatherOutput(BaseModel):
    summary: str
    source: str


class WeatherTool(BaseTool[WeatherInput, WeatherOutput]):
    name = "weather"
    description = "Returns weather summary (offline-safe fallback)"
    input_model = WeatherInput
    output_model = WeatherOutput

    async def run(self, payload: WeatherInput) -> WeatherOutput:
        return WeatherOutput(
            summary=f"Live weather unavailable in offline mode for {payload.location}",
            source="offline-fallback",
        )
