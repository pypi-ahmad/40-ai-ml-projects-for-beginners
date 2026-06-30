"""FastAPI dependency providers."""

from __future__ import annotations

from fastapi import Request

from ml_api.core.config import Settings
from ml_api.core.metrics import MetricsStore
from ml_api.services.model_service import ModelService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service


def get_metrics_store(request: Request) -> MetricsStore:
    return request.app.state.metrics_store
