"""Currency converter with static fallback rates."""

from __future__ import annotations

from pydantic import BaseModel

from crew_platform.tools.base import BaseTool


class CurrencyInput(BaseModel):
    amount: float
    from_currency: str
    to_currency: str


class CurrencyOutput(BaseModel):
    converted_amount: float
    rate: float


_FALLBACK_RATES_USD = {
    "USD": 1.0,
    "EUR": 0.93,
    "INR": 83.5,
    "GBP": 0.79,
    "JPY": 160.0,
}


class CurrencyConverterTool(BaseTool[CurrencyInput, CurrencyOutput]):
    name = "currency_converter"
    description = "Converts currencies using fallback daily rates"
    input_model = CurrencyInput
    output_model = CurrencyOutput

    async def run(self, payload: CurrencyInput) -> CurrencyOutput:
        base = _FALLBACK_RATES_USD.get(payload.from_currency.upper())
        target = _FALLBACK_RATES_USD.get(payload.to_currency.upper())
        if base is None or target is None:
            raise ValueError("Unsupported currency")
        usd_amount = payload.amount / base
        converted = usd_amount * target
        rate = target / base
        return CurrencyOutput(converted_amount=converted, rate=rate)
