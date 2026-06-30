#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv_cache}"

echo "[1/3] Running tests"
uv run pytest -q

echo "[2/3] Running benchmark"
uv run python scripts/run_benchmark.py --runs 3 --prompt-mode medium

echo "[3/3] Executing notebooks"
bash scripts/execute_notebooks.sh

echo "Pipeline completed successfully."
