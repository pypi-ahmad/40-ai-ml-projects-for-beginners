#!/usr/bin/env bash
set -euo pipefail

uv run reasoning-agent benchmark --output-dir artifacts/reports
