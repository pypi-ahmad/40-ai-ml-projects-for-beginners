"""Project constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_NAME = "AI Spreadsheet Analytics Platform"
DEFAULT_ENCODING = "utf-8"

DEFAULT_SCOPES = (
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
)

DEFAULT_WRITE_SCOPES = (
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = ROOT_DIR / "data" / "cache"
DEFAULT_REPORT_DIR = ROOT_DIR / "data" / "reports"
DEFAULT_ARTIFACT_DIR = ROOT_DIR / "data" / "artifacts"
DEFAULT_STATE_DB = DEFAULT_ARTIFACT_DIR / "state.db"

DEFAULT_MAX_SUMMARY_WORDS = 180
DEFAULT_CHAT_HISTORY_LIMIT = 20

QUALITY_CHECKS = (
    "missing_values",
    "duplicate_rows",
    "invalid_dates",
    "invalid_numbers",
    "outliers",
    "mixed_types",
    "empty_columns",
    "constant_columns",
)
