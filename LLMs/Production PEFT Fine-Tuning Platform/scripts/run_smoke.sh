#!/usr/bin/env bash
set -euo pipefail

uv run peft-platform runtime
uv run peft-platform dataset-smoke
uv run peft-platform train --model tinyllama_1_1b_chat --method lora --steps 12
uv run peft-platform benchmark
uv run pytest -q
