#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

uv venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv sync

echo "Environment ready."
