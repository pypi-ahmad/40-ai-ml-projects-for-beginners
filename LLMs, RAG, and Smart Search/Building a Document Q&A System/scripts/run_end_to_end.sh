#!/usr/bin/env bash
set -euo pipefail

PROFILE=${1:-full}
EVAL_PATH=${2:-data/eval/retrieval_eval.jsonl}

uv run python -m local_rag bootstrap
uv run python -m local_rag ingest --profile "$PROFILE"
uv run python -m local_rag evaluate --profile "$PROFILE" --eval-path "$EVAL_PATH"
uv run python -m local_rag failures
uv run python -m local_rag benchmark --profile "$PROFILE"
uv run python -m local_rag diagram

printf '\nEnd-to-end pipeline finished.\n'
