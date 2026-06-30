"""Global exception handlers with consistent error envelopes."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException


def request_id_from_request(request: Request) -> str | None:
    """Return correlation ID attached by request ID middleware."""
    return getattr(request.state, "request_id", None)


def build_error_payload(
    *,
    code: str,
    detail: str,
    request_id: str | None,
    field_errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Construct standard API error payload."""
    payload: dict[str, Any] = {
        "code": code,
        "detail": detail,
        "request_id": request_id,
    }
    if field_errors:
        payload["field_errors"] = field_errors
    return payload


def register_handlers(app: FastAPI) -> None:
    """Register all global exception handlers."""

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        flattened: list[dict[str, str]] = []
        for err in exc.errors():
            loc = err.get("loc", [])
            field_name = ".".join(str(x) for x in loc if x != "body")
            flattened.append(
                {
                    "field": field_name or "body",
                    "message": str(err.get("msg", "Invalid value")),
                }
            )

        return JSONResponse(
            status_code=422,
            content=build_error_payload(
                code="VALIDATION_ERROR",
                detail="Request validation failed.",
                request_id=request_id_from_request(request),
                field_errors=flattened,
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=build_error_payload(
                code=f"HTTP_{exc.status_code}",
                detail=detail,
                request_id=request_id_from_request(request),
            ),
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.opt(exception=exc).error("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content=build_error_payload(
                code="INTERNAL_SERVER_ERROR",
                detail="Internal server error.",
                request_id=request_id_from_request(request),
            ),
        )
