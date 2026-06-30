#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.tmp/matplotlib}"
export AIRFLOW_HOME="${AIRFLOW_HOME:-$PROJECT_ROOT/airflow_config}"
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_ROOT/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$UV_CACHE_DIR"
mkdir -p "$PROJECT_ROOT/.tmp"

if [[ ! -d .venv ]]; then
  uv venv .venv
fi

# Sync can fail in offline mode; keep fallback.
if ! uv sync; then
  echo "uv sync failed (likely offline). Using existing .venv packages."
fi

if .venv/bin/airflow db migrate; then
  echo "Airflow DB migrated"
else
  .venv/bin/airflow db init
fi

.venv/bin/python - <<'PY'
import importlib

checks = [
    "airflow",
    "pandas",
    "numpy",
    "sklearn",
    "joblib",
    "lazypredict",
    "pytest",
    "nbconvert",
]
for module in checks:
    try:
        importlib.import_module(module)
        print(f"[ok] {module}")
    except Exception as exc:
        print(f"[warn] {module} unavailable: {exc.__class__.__name__}: {exc}")

try:
    from flaml import AutoML  # noqa: F401
    print("[ok] flaml.AutoML")
except Exception as exc:
    print(f"[warn] flaml.AutoML unavailable: {exc.__class__.__name__}: {exc}")

try:
    from pycaret.regression import setup  # noqa: F401
    print("[ok] pycaret.regression")
except Exception as exc:
    print(f"[warn] pycaret.regression unavailable: {exc.__class__.__name__}: {exc}")
PY

echo "Bootstrap complete"
