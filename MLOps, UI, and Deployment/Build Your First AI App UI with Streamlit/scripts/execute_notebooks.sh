#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv_cache}"

for nb in notebooks/*.ipynb; do
  echo "[execute] $nb"
  uv run jupyter nbconvert --to notebook --execute "$nb" --inplace --ExecutePreprocessor.timeout=1200
done

echo "Notebook execution complete."
