# Stage 1 — Build environment with uv
FROM python:3.12-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
RUN uv python pin 3.12

COPY pyproject.toml ./
COPY uv.lock ./
COPY app/ app/
COPY src/ src/
COPY models/ models/

RUN uv sync --frozen --no-dev

# Stage 2 — Runtime image
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app /app/app
COPY --from=builder /app/src /app/src
COPY --from=builder /app/models /app/models

RUN addgroup --system app && adduser --system --ingroup app app \
    && mkdir -p /app/logs /app/artifacts/performance \
    && chown -R app:app /app

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

USER app

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
