#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/project.yaml}"
export PYTHONPATH="${PYTHONPATH:-}:src"

if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run dependency install first." >&2
  exit 1
fi

source .venv/bin/activate

export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export LLMFT_TRAIN_MODEL="${LLMFT_TRAIN_MODEL:-sshleifer/tiny-gpt2}"
export LLMFT_TRANSFORMERS_MODEL="${LLMFT_TRANSFORMERS_MODEL:-sshleifer/tiny-gpt2}"

mkdir -p artifacts/logs artifacts/reports "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

python -m llmft --config "$CONFIG_PATH" env validate
python -m llmft --config "$CONFIG_PATH" data build
python -m llmft --config "$CONFIG_PATH" train sft
python -m llmft --config "$CONFIG_PATH" eval run
python -m llmft --config "$CONFIG_PATH" bench run
python -m llmft --config "$CONFIG_PATH" infer run --backend transformers --prompt "Explain QLoRA briefly."
python -m llmft --config "$CONFIG_PATH" export run

python -m llmft --config "$CONFIG_PATH" serve api > artifacts/logs/fastapi_live.log 2>&1 &
API_PID=$!
sleep 8
curl -sSf http://127.0.0.1:8080/health > artifacts/reports/fastapi_health.json
curl -sSf -X POST http://127.0.0.1:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Say hello from live API"}' > artifacts/reports/fastapi_chat.json
kill "$API_PID"
wait "$API_PID" || true

if command -v streamlit >/dev/null 2>&1; then
  timeout 25s streamlit run src/llmft/ui/streamlit_launcher.py \
    --server.headless true \
    --server.port 8501 \
    > artifacts/logs/streamlit_live.log 2>&1 || true
  if ! rg -q "Local URL|Network URL" artifacts/logs/streamlit_live.log; then
    echo "Streamlit launch did not report URL" >&2
    exit 1
  fi
else
  echo "streamlit command missing" >&2
  exit 1
fi
