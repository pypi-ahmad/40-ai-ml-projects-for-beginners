"""FastAPI application factory and middleware/route composition."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import explain, health, info, metrics, predict
from app.services import tracking
from app.services.metrics_store import metrics_store
from app.utils.exceptions import register_handlers
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

logger = setup_logging()


def _parse_origins(raw: str) -> list[str]:
    if raw.strip() == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize tracking and metrics storage on startup."""
    logger.info(
        "Starting {app_name} v{version}",
        app_name=settings.app_name,
        version=settings.app_version,
    )
    app.state.logger = logger

    metrics_store.init()
    tracking.init()

    yield

    logger.info("Shutting down {app_name}", app_name=settings.app_name)


def create_app() -> FastAPI:
    """Construct FastAPI app with middleware stack and routers."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-oriented ML model serving API built with FastAPI, "
            "Pydantic validation, observability middleware, and SHAP explanations."
        ),
        lifespan=lifespan,
    )

    origins = _parse_origins(settings.cors_origins)
    allow_credentials = origins != ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Starlette applies middleware in reverse order of registration.
    # Register inner -> outer so request IDs/security headers also cover early middleware rejections.
    app.add_middleware(RequestSizeLimitMiddleware)
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=settings.rate_limit_per_minute,
        window_seconds=60,
    )
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)

    register_handlers(app)

    app.include_router(health.router)
    app.include_router(info.router)
    app.include_router(predict.router)
    app.include_router(explain.router)
    app.include_router(metrics.router)

    return app


app = create_app()
