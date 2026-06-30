"""API route handlers."""
from __future__ import annotations

from app.routers.explain import router as explain_router
from app.routers.health import router as health_router
from app.routers.info import router as info_router
from app.routers.metrics import router as metrics_router
from app.routers.predict import router as predict_router

__all__ = [
    "explain_router",
    "health_router",
    "info_router",
    "metrics_router",
    "predict_router",
]
