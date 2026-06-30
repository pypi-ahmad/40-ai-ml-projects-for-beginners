"""Input loaders for supported productivity data sources."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_plain_text(text: str) -> str:
    return text.strip()


def load_markdown(markdown: str) -> str:
    return markdown.strip()


def load_json_payload(payload: str | dict[str, Any] | list[Any]) -> str:
    if isinstance(payload, str):
        data: Any = json.loads(payload)
    else:
        data = payload
    return json.dumps(data, ensure_ascii=False, indent=2)


def load_csv_payload(payload: str) -> str:
    rows: list[dict[str, str]] = []
    reader = csv.DictReader(payload.splitlines())
    for row in reader:
        rows.append({k: v for k, v in row.items() if k})
    return json.dumps(rows, ensure_ascii=False, indent=2)


def load_local_file(path: str | Path) -> str:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    content = file_path.read_text(encoding="utf-8")
    if suffix == ".json":
        return load_json_payload(content)
    if suffix == ".csv":
        return load_csv_payload(content)
    return content
