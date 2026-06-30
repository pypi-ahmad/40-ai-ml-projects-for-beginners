"""Project constants shared across modules."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

APP_NAME = "AI SQL Analytics Assistant"
APP_VERSION = "0.1.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
SQLITE_DIR = DATA_DIR / "sqlite"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
DIAGRAMS_DIR = ARTIFACTS_DIR / "diagrams"
SCREENSHOTS_DIR = ARTIFACTS_DIR / "screenshots"
BENCHMARK_DIR = REPO_ROOT / "benchmarks"
CONFIGS_DIR = REPO_ROOT / "configs"

NORTHWIND_DB_PATH = SQLITE_DIR / "northwind_raw.db"
NORTHWIND_SCALED_DB_PATH = SQLITE_DIR / "northwind_scaled.db"
APP_STATE_DB_PATH = SQLITE_DIR / "app_state.db"

DEFAULT_MODEL = "qwen3.5:4b"
COMPARISON_MODEL = "granite4.1:3b"
JUDGE_MODEL = "granite4.1:3b"
OLLAMA_HOST = "http://127.0.0.1:11434"

BLOCKED_SQL_KEYWORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "create",
    "attach",
    "detach",
    "replace",
    "pragma",
    "vacuum",
    "reindex",
}

SUPPORTED_CHARTS = [
    "table",
    "bar",
    "line",
    "pie",
    "scatter",
    "histogram",
    "heatmap",
    "time_series",
]
