#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/baseline.yaml}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"

.venv/bin/domain-llm train --config-path "$CONFIG_PATH"
.venv/bin/domain-llm benchmark --config-path "$CONFIG_PATH"
.venv/bin/domain-llm export --config-path "$CONFIG_PATH"
.venv/bin/domain-llm collect-evidence --config-path "$CONFIG_PATH"
