"""Rate limiting configuration for production-safe inference endpoints."""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import FastAPI


def configure_rate_limiting(app: FastAPI, default_limit: str = "100/minute"):
    """Attach SlowAPI limiter and exception handler to FastAPI app."""
    limiter = Limiter(key_func=get_remote_address, default_limits=[default_limit])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    return limiter
