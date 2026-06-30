"""FastAPI application factory for Ames Housing model serving."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ml_api.api.routes import router as api_router
from ml_api.core.config import Settings, get_settings
from ml_api.core.errors import APIError, PayloadTooLargeError
from ml_api.core.logging import configure_logging
from ml_api.core.metrics import MetricsStore
from ml_api.schemas.common import ErrorPayload, ErrorResponse
from ml_api.services.model_service import ModelService

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    configure_logging()
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model_service.load()
        logger.info("FastAPI startup complete. Model ready=%s", app.state.model_service.is_ready)
        yield
        logger.info("FastAPI shutdown complete")

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-grade regression API for curated Ames Housing data, "
            "including validation, batch inference, metrics, and explainability."
        ),
        lifespan=lifespan,
    )

    app.state.settings = settings
    app.state.metrics_store = MetricsStore()
    app.state.model_service = ModelService(settings)
    app.state.started_at = time.time()

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        started = time.perf_counter()

        raw_len_header = request.headers.get("content-length")
        if raw_len_header and raw_len_header.isdigit():
            content_len = int(raw_len_header)
            if content_len > settings.max_request_bytes:
                error = PayloadTooLargeError(settings.max_request_bytes, content_len)
                return _error_response(error, request_id=request.state.request_id)

        response = await call_next(request)
        latency_ms = (time.perf_counter() - started) * 1000
        app.state.metrics_store.record(
            route=request.url.path,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )
        response.headers["x-request-id"] = request.state.request_id
        return response

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        request_id = getattr(request.state, "request_id", "unknown")
        return _error_response(exc, request_id=request_id)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", "unknown")
        details = {"validation": str(exc.errors()[:5])}
        payload = ErrorResponse(
            error=ErrorPayload(
                code="VALIDATION_ERROR",
                message="Request validation failed",
                details=details,
                request_id=request_id,
            )
        )
        return JSONResponse(status_code=422, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):  # pragma: no cover
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception("Unhandled error: %s", exc)
        payload = ErrorResponse(
            error=ErrorPayload(
                code="INTERNAL_SERVER_ERROR",
                message="Unexpected server error",
                details={},
                request_id=request_id,
            )
        )
        return JSONResponse(status_code=500, content=payload.model_dump())

    app.include_router(api_router)
    return app


app = create_app()


def _error_response(exc: APIError, request_id: str) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorPayload(
            code=exc.code,
            message=exc.message,
            details=exc.details,
            request_id=request_id,
        )
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())
