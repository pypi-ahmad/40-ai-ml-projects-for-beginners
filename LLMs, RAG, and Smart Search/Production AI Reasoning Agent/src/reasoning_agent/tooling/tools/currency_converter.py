"""Currency conversion tool with pluggable provider fallback."""

from __future__ import annotations

import httpx
from pydantic import BaseModel, Field

from reasoning_agent.tooling.base import ToolContext, ToolSpec


class CurrencyInput(BaseModel):
    """Currency conversion input payload."""

    amount: float
    from_currency: str = Field(min_length=3, max_length=3)
    to_currency: str = Field(min_length=3, max_length=3)


class CurrencyOutput(BaseModel):
    """Currency conversion output payload."""

    converted_amount: float | None = None
    rate: float | None = None
    provider: str
    available: bool
    message: str


class CurrencyProvider:
    """Provider adapter for currency conversion."""

    def __init__(self, provider: str = "frankfurter", api_key: str = "") -> None:
        self.provider = provider
        self.api_key = api_key

    def convert(self, amount: float, source: str, target: str) -> CurrencyOutput:
        """Convert with configured provider."""

        source = source.upper()
        target = target.upper()

        if self.provider == "frankfurter":
            return self._frankfurter(amount, source, target)

        return CurrencyOutput(
            provider=self.provider,
            available=False,
            message="Unsupported currency provider",
        )

    def _frankfurter(self, amount: float, source: str, target: str) -> CurrencyOutput:
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(
                    "https://api.frankfurter.app/latest",
                    params={"amount": amount, "from": source, "to": target},
                )
                resp.raise_for_status()
                payload = resp.json()
            rates = payload.get("rates", {})
            converted = float(rates[target])
            rate = converted / amount if amount != 0 else 0.0
            return CurrencyOutput(
                converted_amount=converted,
                rate=rate,
                provider="frankfurter",
                available=True,
                message="ok",
            )
        except Exception as exc:
            return CurrencyOutput(
                provider="frankfurter",
                available=False,
                message=f"currency provider unavailable: {exc}",
            )


def make_handler(provider: CurrencyProvider):
    """Create currency tool handler."""

    def handler(payload: CurrencyInput, _: ToolContext) -> CurrencyOutput:
        return provider.convert(payload.amount, payload.from_currency, payload.to_currency)

    return handler


spec = ToolSpec(
    name="currency_converter",
    description="Convert currency amounts with provider adapter and graceful fallback",
    input_model=CurrencyInput,
    output_model=CurrencyOutput,
    tags=["finance", "conversion"],
    requires_network=True,
)
