#!/usr/bin/env bash
set -euo pipefail

uv python install 3.12.10
uv sync --all-extras

cp -n .env.example .env || true

echo "Setup complete."
echo "Run: uv run streamlit run streamlit_app/Home.py"
