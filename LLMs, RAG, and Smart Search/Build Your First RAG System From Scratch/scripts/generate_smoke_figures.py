"""Generate figure artifacts from smoke evaluation outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rag_system.visualization import generate_all_core_visuals


def main() -> None:
    base = Path("data/artifacts/smoke")
    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    docs = pd.read_parquet("data/processed/documents.parquet")
    hallucination = pd.read_parquet(base / "tables" / "hallucination.parquet")

    generate_all_core_visuals(
        doc_df=docs,
        retrieval_summary=summary["retrieval"],
        generation_summary=summary["generation"],
        hallucination_df=hallucination,
        output_dir=base / "figures",
    )


if __name__ == "__main__":
    main()
