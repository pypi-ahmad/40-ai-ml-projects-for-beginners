#!/usr/bin/env bash
set -euo pipefail

PROMPT="${1:-Calculate 12 * 7 and explain reasoning.}"
uv run reasoning-agent chat "$PROMPT" --session-id cli-demo --json
