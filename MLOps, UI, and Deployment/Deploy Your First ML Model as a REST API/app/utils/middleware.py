"""Custom ASGI middleware for security, observability, and platform controls."""
from __future__ import annotations

import hmac
import random
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.config import settings
from app.services.metrics_store import RequestMetricEvent, metrics_store
from app.utils.exceptions import build_error_payload


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach standard secure-by-default headers to each response."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject per-request correlation ID into state and response header."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests that exceed configured content-length size limit."""

    _BODY_METHODS = {"POST", "PUT", "PATCH"}

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                content_length_int = int(content_length)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content=build_error_payload(
                        code="INVALID_CONTENT_LENGTH",
                        detail="Invalid content-length header.",
                        request_id=getattr(request.state, "request_id", None),
                    ),
                )

            if content_length_int < 0:
                return JSONResponse(
                    status_code=400,
                    content=build_error_payload(
                        code="INVALID_CONTENT_LENGTH",
                        detail="content-length must be non-negative.",
                        request_id=getattr(request.state, "request_id", None),
                    ),
                )

            if content_length_int > settings.max_request_body_bytes:
                return JSONResponse(
                    status_code=413,
                    content=build_error_payload(
                        code="REQUEST_TOO_LARGE",
                        detail=(
                            f"Request body exceeds max_request_body_bytes="
                            f"{settings.max_request_body_bytes}."
                        ),
                        request_id=getattr(request.state, "request_id", None),
                    ),
                )

        if request.method in self._BODY_METHODS:
            # Validate actual body size as defense-in-depth for missing/incorrect content-length headers.
            body = await request.body()
            if len(body) > settings.max_request_body_bytes:
                return JSONResponse(
                    status_code=413,
                    content=build_error_payload(
                        code="REQUEST_TOO_LARGE",
                        detail=(
                            f"Request body exceeds max_request_body_bytes="
                            f"{settings.max_request_body_bytes}."
                        ),
                        request_id=getattr(request.state, "request_id", None),
                    ),
                )

        return await call_next(request)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Protect selected routes with optional API key auth."""

    _PROTECTED_PATHS = {"/predict", "/predict-batch", "/explain", "/admin/reload"}

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if not settings.api_key:
            return await call_next(request)

        if request.url.path not in self._PROTECTED_PATHS:
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(provided, settings.api_key):
            return JSONResponse(
                status_code=401,
                content=build_error_payload(
                    code="UNAUTHORIZED",
                    detail="Invalid or missing API key.",
                    request_id=getattr(request.state, "request_id", None),
                ),
            )

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory per-IP limiter suitable for local/demo environments."""

    def __init__(self, app: ASGIApp, max_requests: int = 60, window_seconds: int = 60) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._history: dict[str, list[float]] = defaultdict(list)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        client_ip: str = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window_start = now - self.window_seconds

        history = self._history[client_ip]
        while history and history[0] < window_start:
            history.pop(0)

        if len(history) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content=build_error_payload(
                    code="RATE_LIMIT_EXCEEDED",
                    detail="Rate limit exceeded. Try again later.",
                    request_id=getattr(request.state, "request_id", None),
                ),
            )

        history.append(now)
        return await call_next(request)


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Structured access log writer with request duration."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000.0
        request.state.latency_ms = duration_ms

        if random.random() <= settings.request_log_sample_rate:
            logger.info(
                "{method} {path} -> {status} ({duration:.2f}ms) request_id={request_id}",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration=duration_ms,
                request_id=getattr(request.state, "request_id", "-"),
            )
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Persist per-request metrics into SQLite backend."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = getattr(request.state, "latency_ms", (time.perf_counter() - start) * 1000.0)

        try:
            metrics_store.record(
                RequestMetricEvent(
                    timestamp_utc=datetime.now(UTC).isoformat(),
                    endpoint=request.url.path,
                    method=request.method,
                    status_code=response.status_code,
                    latency_ms=float(duration_ms),
                    request_id=getattr(request.state, "request_id", ""),
                )
            )
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.warning("Failed to record request metrics: {err}", err=exc)

        return response
