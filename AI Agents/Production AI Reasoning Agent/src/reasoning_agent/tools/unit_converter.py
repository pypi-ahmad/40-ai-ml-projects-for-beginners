"""Simple unit converter tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from reasoning_agent.tools.base import BaseTool


class UnitInput(BaseModel):
    value: float
    from_unit: str = Field(min_length=1)
    to_unit: str = Field(min_length=1)


class UnitOutput(BaseModel):
    converted_value: float


_UNIT_FACTORS = {
    "m": 1.0,
    "km": 1000.0,
    "cm": 0.01,
    "mm": 0.001,
    "mi": 1609.344,
    "ft": 0.3048,
    "in": 0.0254,
}


class UnitConverterTool(BaseTool[UnitInput, UnitOutput]):
    name = "unit_converter"
    description = "Converts supported length units"
    input_model = UnitInput
    output_model = UnitOutput

    async def run(self, payload: UnitInput) -> UnitOutput:
        from_factor = _UNIT_FACTORS.get(payload.from_unit.lower())
        to_factor = _UNIT_FACTORS.get(payload.to_unit.lower())
        if from_factor is None or to_factor is None:
            raise ValueError("Unsupported unit")
        meters = payload.value * from_factor
        return UnitOutput(converted_value=meters / to_factor)
