"""Weather tool with pluggable provider and graceful fallback."""

from __future__ import annotations

import httpx
from pydantic import BaseModel

from reasoning_agent.tooling.base import ToolContext, ToolSpec


class WeatherInput(BaseModel):
    """Weather input payload."""

    location: str


class WeatherOutput(BaseModel):
    """Weather output payload."""

    provider: str
    available: bool
    summary: str
    temperature_c: float | None = None
    wind_kmh: float | None = None


class WeatherProvider:
    """Weather provider adapter."""

    def __init__(self, provider: str = "open_meteo", api_key: str = "") -> None:
        self.provider = provider
        self.api_key = api_key

    def weather(self, location: str) -> WeatherOutput:
        """Get weather for location."""

        if self.provider == "open_meteo":
            return self._open_meteo(location)

        return WeatherOutput(
            provider=self.provider,
            available=False,
            summary="unsupported weather provider",
        )

    def _open_meteo(self, location: str) -> WeatherOutput:
        try:
            with httpx.Client(timeout=15.0) as client:
                geo = client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": location, "count": 1},
                )
                geo.raise_for_status()
                g_payload = geo.json()
                results = g_payload.get("results", [])
                if not results:
                    return WeatherOutput(provider="open_meteo", available=False, summary="location not found")

                lat = results[0]["latitude"]
                lon = results[0]["longitude"]
                weather = client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current": "temperature_2m,wind_speed_10m",
                    },
                )
                weather.raise_for_status()
                w_payload = weather.json().get("current", {})

            temp = float(w_payload.get("temperature_2m", 0.0))
            wind = float(w_payload.get("wind_speed_10m", 0.0))
            return WeatherOutput(
                provider="open_meteo",
                available=True,
                summary=f"{location}: {temp:.1f}C, wind {wind:.1f} km/h",
                temperature_c=temp,
                wind_kmh=wind,
            )
        except Exception as exc:
            return WeatherOutput(
                provider="open_meteo",
                available=False,
                summary=f"weather provider unavailable: {exc}",
            )


def make_handler(provider: WeatherProvider):
    """Create weather tool handler."""

    def handler(payload: WeatherInput, _: ToolContext) -> WeatherOutput:
        return provider.weather(payload.location)

    return handler


spec = ToolSpec(
    name="weather",
    description="Get current weather by location using provider adapter",
    input_model=WeatherInput,
    output_model=WeatherOutput,
    tags=["weather", "search"],
    requires_network=True,
)
