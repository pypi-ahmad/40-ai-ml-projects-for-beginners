"""Custom exception hierarchy for consistent API errors."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class APIError(Exception):
    """Base API exception with HTTP status and machine-readable code."""

    code: str
    message: str
    status_code: int
    details: dict[str, str | int | float] = field(default_factory=dict)


class ModelNotReadyError(APIError):
    def __init__(self, message: str = "Model artifacts are not loaded") -> None:
        super().__init__(
            code="MODEL_NOT_READY",
            message=message,
            status_code=503,
        )


class BatchLimitError(APIError):
    def __init__(self, max_batch_size: int, requested: int) -> None:
        super().__init__(
            code="BATCH_LIMIT_EXCEEDED",
            message=f"Batch size {requested} exceeds max allowed size {max_batch_size}",
            status_code=422,
            details={"max_batch_size": max_batch_size, "requested": requested},
        )


class PayloadTooLargeError(APIError):
    def __init__(self, max_bytes: int, actual_bytes: int) -> None:
        super().__init__(
            code="PAYLOAD_TOO_LARGE",
            message=f"Payload too large ({actual_bytes} bytes). Limit is {max_bytes} bytes.",
            status_code=413,
            details={"max_bytes": max_bytes, "actual_bytes": actual_bytes},
        )
