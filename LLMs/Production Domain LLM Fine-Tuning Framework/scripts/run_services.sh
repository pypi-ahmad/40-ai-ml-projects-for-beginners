#!/usr/bin/env bash
set -euo pipefail

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"

.venv/bin/domain-llm serve-api --host 0.0.0.0 --port 8000 &
API_PID=$!
.venv/bin/domain-llm serve-ui --port 8501 &
UI_PID=$!

cleanup() {
  kill "$API_PID" "$UI_PID"
}
trap cleanup EXIT

wait
