#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="outputs/reports/verification_${TS}"
mkdir -p "$LOG_DIR"
VERIFY_PROFILE="${VERIFY_PROFILE:-quickstart}"

if [[ "$VERIFY_PROFILE" != "full" && "$VERIFY_PROFILE" != "quickstart" ]]; then
  echo "[verify] invalid VERIFY_PROFILE=${VERIFY_PROFILE} (expected: full|quickstart)"
  exit 1
fi

echo "[verify] starting full pipeline verification"

if [[ -x ".venv/bin/python" ]]; then
  PY=(".venv/bin/python")
  STREAMLIT=(".venv/bin/streamlit")
else
  PY=("uv" "run" "python")
  STREAMLIT=("uv" "run" "streamlit")
fi

run_and_log() {
  local name="$1"
  shift
  echo "[verify] $name"
  if ! "$@" >"${LOG_DIR}/${name}.log" 2>&1; then
    echo "[verify] ${name} failed. Last log lines:"
    tail -n 80 "${LOG_DIR}/${name}.log" || true
    return 1
  fi
}

probe_health() {
  local url="$1"
  "${PY[@]}" - "$url" <<'PY'
import json
import sys
from urllib.request import urlopen

url = sys.argv[1]
try:
    with urlopen(url, timeout=2.0) as response:  # noqa: S310
        body = response.read().decode("utf-8", errors="replace")
except Exception:
    raise SystemExit(1)

body_normalized = body.strip().lower()
if "ok" not in body_normalized:
    raise SystemExit(1)

payload = {"status": body.strip(), "url": url}
print(json.dumps(payload))
PY
}

run_and_log validate_local "${PY[@]}" -m local_rag validate-local
if ! run_and_log corpus_report_full "${PY[@]}" -m local_rag corpus-report --profile full; then
  run_and_log bootstrap "${PY[@]}" -m local_rag bootstrap --build-quickstart
  run_and_log corpus_report_full "${PY[@]}" -m local_rag corpus-report --profile full
fi
if ! run_and_log corpus_report_quickstart "${PY[@]}" -m local_rag corpus-report --profile quickstart; then
  run_and_log build_quickstart "${PY[@]}" -m local_rag build-quickstart
  run_and_log corpus_report_quickstart "${PY[@]}" -m local_rag corpus-report --profile quickstart
fi

run_and_log doctor "${PY[@]}" -m local_rag doctor
run_and_log ingest_primary "${PY[@]}" -m local_rag ingest --profile "$VERIFY_PROFILE"
run_and_log validate_index_primary "${PY[@]}" -m local_rag validate-index --profile "$VERIFY_PROFILE"
run_and_log ingest_primary_second "${PY[@]}" -m local_rag ingest --profile "$VERIFY_PROFILE"
run_and_log query_primary "${PY[@]}" -m local_rag query "What is ACPI?" --top-k 5 --profile "$VERIFY_PROFILE"
run_and_log generate_eval "${PY[@]}" -m local_rag generate-eval-set --max-examples 50 --profile "$VERIFY_PROFILE"
run_and_log compile_eval "${PY[@]}" -m local_rag compile-eval-set --include-unverified
run_and_log evaluate "${PY[@]}" -m local_rag evaluate --profile "$VERIFY_PROFILE"
run_and_log run_experiments "${PY[@]}" -m local_rag run-experiments --profile quickstart
run_and_log failures "${PY[@]}" -m local_rag failures

echo "[verify] streamlit real run validation"
PORT=8502
HEALTH_URL="http://127.0.0.1:${PORT}/_stcore/health"
STREAMLIT_LOG="${LOG_DIR}/streamlit_real_run.log"

"${STREAMLIT[@]}" run streamlit_app/app.py --server.headless true --server.port "$PORT" \
  >"$STREAMLIT_LOG" 2>&1 &
STREAMLIT_PID=$!

cleanup() {
  if kill -0 "$STREAMLIT_PID" >/dev/null 2>&1; then
    kill "$STREAMLIT_PID" >/dev/null 2>&1 || true
    wait "$STREAMLIT_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 30); do
  if probe_health "$HEALTH_URL" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [[ "$READY" -ne 1 ]]; then
  echo "[verify] streamlit failed to become healthy, see ${STREAMLIT_LOG}"
  exit 1
fi

echo "[verify] streamlit healthy endpoint reached"
probe_health "$HEALTH_URL" >"${LOG_DIR}/streamlit_health.json"
echo "[verify] streamlit real run succeeded"

echo "[verify] completed. Logs: ${LOG_DIR}"
