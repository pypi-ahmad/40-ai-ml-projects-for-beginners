#!/usr/bin/env bash
set -euo pipefail

uv venv .venv
source .venv/bin/activate
uv sync --all-groups

if [ ! -f .env ]; then
  cp .env.example .env
fi

echo "Setup complete"
