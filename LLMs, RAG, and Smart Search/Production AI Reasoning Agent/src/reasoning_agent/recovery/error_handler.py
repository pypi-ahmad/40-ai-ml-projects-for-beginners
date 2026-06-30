"""Failure handling and retry policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecoveryDecision:
    """Error handling decision."""

    retry: bool
    revised_step: str | None
    reason: str


class ErrorHandler:
    """Apply bounded retries and simple recovery planning."""

    def decide(
        self,
        *,
        error: str,
        retries: int,
        max_retries: int,
        current_step: str,
    ) -> RecoveryDecision:
        """Compute retry decision."""

        if retries >= max_retries:
            return RecoveryDecision(retry=False, revised_step=None, reason="retry_limit")

        lowered = error.lower()
        if "validation" in lowered:
            revised = f"Reformat tool arguments for step: {current_step}"
            return RecoveryDecision(retry=True, revised_step=revised, reason="input_validation")

        if "timeout" in lowered or "unavailable" in lowered:
            revised = f"Try alternate tool or continue without external data for step: {current_step}"
            return RecoveryDecision(retry=True, revised_step=revised, reason="provider_failure")

        return RecoveryDecision(retry=True, revised_step=current_step, reason="generic_retry")
