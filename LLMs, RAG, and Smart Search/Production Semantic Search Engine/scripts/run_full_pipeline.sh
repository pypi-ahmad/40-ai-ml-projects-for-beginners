#!/usr/bin/env bash
set -euo pipefail

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

uv run semantic-search check-models
uv run semantic-search ingest --source huggingface
uv run semantic-search chunk
uv run semantic-search index --model primary
uv run python scripts/generate_eval_queries.py
uv run semantic-search evaluate --mode hybrid
uv run semantic-search benchmark

echo "Pipeline complete"
