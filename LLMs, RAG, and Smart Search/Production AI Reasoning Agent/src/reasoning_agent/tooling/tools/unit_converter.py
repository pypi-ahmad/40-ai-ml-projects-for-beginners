"""Unit conversion tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from reasoning_agent.tooling.base import ToolContext, ToolSpec


class UnitConverterInput(BaseModel):
    """Unit conversion input payload."""

    value: float
    from_unit: str = Field(description="Supported: m, km, cm, kg, g, lb, c, f")
    to_unit: str = Field(description="Supported: m, km, cm, kg, g, lb, c, f")


class UnitConverterOutput(BaseModel):
    """Unit conversion output payload."""

    converted_value: float
    from_unit: str
    to_unit: str


_LINEAR_FACTORS = {
    "m": 1.0,
    "km": 1000.0,
    "cm": 0.01,
    "kg": 1.0,
    "g": 0.001,
    "lb": 0.45359237,
}


def convert_units(payload: UnitConverterInput, _: ToolContext) -> UnitConverterOutput:
    """Convert supported units including temperature."""

    src = payload.from_unit.lower()
    dst = payload.to_unit.lower()
    value = payload.value

    if src in {"c", "f"} or dst in {"c", "f"}:
        if src == "c" and dst == "f":
            out = value * 9 / 5 + 32
        elif src == "f" and dst == "c":
            out = (value - 32) * 5 / 9
        elif src == dst:
            out = value
        else:
            raise ValueError("Invalid temperature conversion")
        return UnitConverterOutput(converted_value=out, from_unit=src, to_unit=dst)

    if src not in _LINEAR_FACTORS or dst not in _LINEAR_FACTORS:
        raise ValueError("Unsupported unit")

    base = value * _LINEAR_FACTORS[src]
    out = base / _LINEAR_FACTORS[dst]
    return UnitConverterOutput(converted_value=out, from_unit=src, to_unit=dst)


spec = ToolSpec(
    name="unit_converter",
    description="Convert common units (distance, mass, temperature)",
    input_model=UnitConverterInput,
    output_model=UnitConverterOutput,
    tags=["math", "utility", "conversion"],
)
