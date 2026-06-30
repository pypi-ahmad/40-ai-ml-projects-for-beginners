"""Utility tools: calculator, unit conversion, datetime, weather, currency."""

from __future__ import annotations

import ast
import math
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from pydantic import BaseModel, Field

from internet_agent.tools.base import BaseTool


class CalculatorInput(BaseModel):
    expression: str


class CalculatorOutput(BaseModel):
    result: float


class CalculatorTool(BaseTool[CalculatorInput, CalculatorOutput]):
    name = "calculator"
    description = "Evaluate numeric expression safely."
    input_model = CalculatorInput
    output_model = CalculatorOutput

    def _safe_eval(self, expression: str) -> float:
        allowed_nodes = {
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.Mod,
            ast.FloorDiv,
            ast.USub,
            ast.UAdd,
        }
        node = ast.parse(expression, mode="eval")
        for child in ast.walk(node):
            if type(child) not in allowed_nodes:  # noqa: E721
                raise ValueError("Unsupported expression")
        return float(eval(compile(node, "<expr>", "eval"), {"__builtins__": {}}, {}))

    async def run(self, payload: CalculatorInput) -> CalculatorOutput:
        return CalculatorOutput(result=self._safe_eval(payload.expression))


class PythonCalculatorInput(BaseModel):
    expression: str


class PythonCalculatorOutput(BaseModel):
    result: str


class PythonCalculatorTool(BaseTool[PythonCalculatorInput, PythonCalculatorOutput]):
    name = "python_calculator"
    description = "Evaluate math-only Python expression in restricted scope."
    input_model = PythonCalculatorInput
    output_model = PythonCalculatorOutput

    async def run(self, payload: PythonCalculatorInput) -> PythonCalculatorOutput:
        safe_globals = {"__builtins__": {}, "math": math}
        safe_locals: dict[str, Any] = {}
        node = ast.parse(payload.expression, mode="eval")
        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom, ast.Call, ast.Attribute, ast.Name)):
                if isinstance(child, ast.Name) and child.id == "math":
                    continue
                if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name) and child.value.id == "math":
                    continue
                if isinstance(child, ast.Call):
                    continue
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    raise ValueError("Imports not allowed")
        result = eval(compile(node, "<pyexpr>", "eval"), safe_globals, safe_locals)
        return PythonCalculatorOutput(result=str(result))


class UnitInput(BaseModel):
    value: float
    from_unit: str
    to_unit: str


class UnitOutput(BaseModel):
    result: float
    formula: str


class UnitConverterTool(BaseTool[UnitInput, UnitOutput]):
    name = "unit_converter"
    description = "Convert between common length, weight, and temperature units."
    input_model = UnitInput
    output_model = UnitOutput

    _linear = {
        "m": 1.0,
        "km": 1000.0,
        "cm": 0.01,
        "mm": 0.001,
        "mi": 1609.34,
        "ft": 0.3048,
        "kg": 1.0,
        "g": 0.001,
        "lb": 0.453592,
    }

    async def run(self, payload: UnitInput) -> UnitOutput:
        from_unit = payload.from_unit.lower()
        to_unit = payload.to_unit.lower()

        if from_unit in {"c", "f"} or to_unit in {"c", "f"}:
            if from_unit == "c" and to_unit == "f":
                result = payload.value * 9 / 5 + 32
                return UnitOutput(result=result, formula="C * 9/5 + 32")
            if from_unit == "f" and to_unit == "c":
                result = (payload.value - 32) * 5 / 9
                return UnitOutput(result=result, formula="(F - 32) * 5/9")
            raise ValueError("Temperature conversion supports only C<->F")

        if from_unit not in self._linear or to_unit not in self._linear:
            raise ValueError("Unsupported units")

        base = payload.value * self._linear[from_unit]
        result = base / self._linear[to_unit]
        return UnitOutput(result=result, formula=f"{payload.value}*{from_unit}->{to_unit}")


class DateTimeInput(BaseModel):
    timezone_offset: str = Field(default="+00:00", description="Example: +05:30")


class DateTimeOutput(BaseModel):
    iso_time: str


class DateTimeTool(BaseTool[DateTimeInput, DateTimeOutput]):
    name = "date_time"
    description = "Get current date and time in given UTC offset."
    input_model = DateTimeInput
    output_model = DateTimeOutput

    async def run(self, payload: DateTimeInput) -> DateTimeOutput:
        sign = 1 if payload.timezone_offset.startswith("+") else -1
        hours, minutes = payload.timezone_offset[1:].split(":")
        delta_seconds = sign * (int(hours) * 3600 + int(minutes) * 60)
        tz = timezone(timedelta(seconds=delta_seconds))
        now = datetime.now(tz=tz)
        return DateTimeOutput(iso_time=now.isoformat())


class WeatherInput(BaseModel):
    location: str


class WeatherOutput(BaseModel):
    location: str
    temperature_c: float
    wind_kmh: float
    weather_code: int


class WeatherTool(BaseTool[WeatherInput, WeatherOutput]):
    name = "weather"
    description = "Get current weather using Open-Meteo (no-key default)."
    input_model = WeatherInput
    output_model = WeatherOutput

    async def run(self, payload: WeatherInput) -> WeatherOutput:
        async with httpx.AsyncClient(timeout=20) as client:
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": payload.location, "count": 1, "language": "en", "format": "json"},
            )
            geo.raise_for_status()
            geo_data = geo.json()
            if not geo_data.get("results"):
                raise ValueError("Location not found")
            first = geo_data["results"][0]
            weather = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": first["latitude"],
                    "longitude": first["longitude"],
                    "current": "temperature_2m,wind_speed_10m,weather_code",
                },
            )
            weather.raise_for_status()
            current = weather.json().get("current", {})

        return WeatherOutput(
            location=f"{first['name']}, {first.get('country', '')}".strip(", "),
            temperature_c=float(current.get("temperature_2m", 0.0)),
            wind_kmh=float(current.get("wind_speed_10m", 0.0)),
            weather_code=int(current.get("weather_code", 0)),
        )


class CurrencyInput(BaseModel):
    amount: float = 1.0
    from_currency: str
    to_currency: str


class CurrencyOutput(BaseModel):
    converted_amount: float
    rate: float
    as_of: str


class CurrencyExchangeTool(BaseTool[CurrencyInput, CurrencyOutput]):
    name = "currency_exchange"
    description = "Convert currencies using Frankfurter API."
    input_model = CurrencyInput
    output_model = CurrencyOutput

    async def run(self, payload: CurrencyInput) -> CurrencyOutput:
        base = payload.from_currency.upper()
        target = payload.to_currency.upper()

        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                f"https://api.frankfurter.app/latest?from={base}&to={target}"
            )
            response.raise_for_status()
            data = response.json()

        rate = float(data["rates"][target])
        return CurrencyOutput(
            converted_amount=payload.amount * rate,
            rate=rate,
            as_of=data.get("date", ""),
        )
