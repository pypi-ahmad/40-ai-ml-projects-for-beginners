"""Utility modules."""
from __future__ import annotations

from app.utils.exceptions import build_error_payload, register_handlers, request_id_from_request
from app.utils.logging import setup_logging
from app.utils.middleware import (
    APIKeyMiddleware,
    AccessLogMiddleware,
    MetricsMiddleware,
    RateLimitMiddleware,
    RequestIDMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)

__all__ = [
    "APIKeyMiddleware",
    "AccessLogMiddleware",
    "MetricsMiddleware",
    "RateLimitMiddleware",
    "RequestIDMiddleware",
    "RequestSizeLimitMiddleware",
    "SecurityHeadersMiddleware",
    "build_error_payload",
    "register_handlers",
    "request_id_from_request",
    "setup_logging",
]
