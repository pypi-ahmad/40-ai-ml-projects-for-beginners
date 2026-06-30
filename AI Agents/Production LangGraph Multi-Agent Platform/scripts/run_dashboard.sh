#!/usr/bin/env bash
set -euo pipefail
uv run streamlit run apps/streamlit/app.py --server.port 8501 --server.address 0.0.0.0
