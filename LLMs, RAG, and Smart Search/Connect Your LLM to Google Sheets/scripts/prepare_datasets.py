"""Prepare local realistic datasets for Google Sheets upload."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - fallback for bootstrap env.
    pd = None  # type: ignore[assignment]


def profile_dataset(path: Path) -> dict:
    if pd is not None:
        df = pd.read_csv(path)
        return {
            "name": path.stem,
            "path": str(path),
            "rows": len(df),
            "columns": len(df.columns),
            "dtypes": {k: str(v) for k, v in df.dtypes.items()},
            "missing": df.isna().sum().astype(int).to_dict(),
        }

    import csv

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames or []
    missing = {col: sum(1 for row in rows if row.get(col) in {"", None}) for col in columns}
    return {
        "name": path.stem,
        "path": str(path),
        "rows": len(rows),
        "columns": len(columns),
        "dtypes": {},
        "missing": missing,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    samples_dir = root / "data" / "samples"
    artifact_dir = root / "data" / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    generated_at = pd.Timestamp.now("UTC").isoformat() if pd is not None else datetime.now(UTC).isoformat()
    catalog = {"generated_at": generated_at, "datasets": [profile_dataset(path) for path in sorted(samples_dir.glob("*.csv"))]}

    catalog_path = artifact_dir / "dataset_catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"Wrote {catalog_path}")


if __name__ == "__main__":
    main()
