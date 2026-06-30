FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    MPLCONFIGDIR=/app/.mplconfig

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md ./
COPY ml_package ./ml_package
COPY api ./api
COPY scripts ./scripts
COPY tests ./tests
COPY models ./models
COPY outputs ./outputs
COPY notebooks ./notebooks
COPY cli ./cli

RUN mkdir -p /app/.mplconfig /app/logs
RUN uv sync --frozen || uv sync

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
