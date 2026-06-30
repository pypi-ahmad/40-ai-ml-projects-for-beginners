#!/usr/bin/env bash
set -euo pipefail

# End-to-end automation entrypoint for local reproducible project execution.
PROFILE="${1:-balanced}" # fast|balanced|deep
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "[1/8] Sync dependencies with uv"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv sync

echo "[2/8] Train and benchmark models (${PROFILE} profile)"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv run python scripts/train_pipeline.py --profile "${PROFILE}"

echo "[3/8] Run API tests"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv run pytest -q

echo "[4/8] Build optimized Docker image"
docker build -f docker/Dockerfile.optimized -t ml-api:latest .

echo "[5/8] Start compose stack (api + prometheus + grafana)"
# Raise rate limits during load benchmarking to avoid synthetic 429 failures.
RATE_LIMIT_DEFAULT="${RATE_LIMIT_DEFAULT:-20000/minute}" \
RATE_LIMIT_PREDICT="${RATE_LIMIT_PREDICT:-20000/minute}" \
RATE_LIMIT_BATCH="${RATE_LIMIT_BATCH:-2000/minute}" \
RATE_LIMIT_EXPLAIN="${RATE_LIMIT_EXPLAIN:-2000/minute}" \
docker compose up -d --build

cleanup() {
  echo "[8/8] Tear down compose stack"
  docker compose down
}
trap cleanup EXIT

echo "[6/8] Smoke test deployed API"
curl -fsS "http://127.0.0.1:8001/health" >/dev/null
curl -fsS "http://127.0.0.1:8001/model-info" >/dev/null

echo "[7/8] Capture Docker-side benchmark"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv run python scripts/performance_benchmark.py \
  --label docker \
  --base-url "http://127.0.0.1:8001" \
  --requests 120 \
  --concurrency 20

echo "Pipeline complete."
