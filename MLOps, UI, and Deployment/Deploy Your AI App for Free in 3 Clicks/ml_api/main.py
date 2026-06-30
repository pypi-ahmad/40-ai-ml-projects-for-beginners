"""ASGI entrypoint module for uvicorn."""

from ml_api.app import app

__all__ = ["app"]
