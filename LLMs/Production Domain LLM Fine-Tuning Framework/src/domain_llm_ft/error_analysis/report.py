"""Error analysis report generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from domain_llm_ft.utils.io import write_json


def generate_error_analysis(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    text_column: str,
    output_dir: Path,
) -> dict[str, int]:
    """Generate structured error analysis artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    working = frame.copy()
    working["y_true"] = y_true
    working["y_pred"] = y_pred
    working["confidence"] = confidences
    working["is_misclassified"] = working["y_true"] != working["y_pred"]
    working["text_len"] = working[text_column].astype(str).str.len()

    misclassified = working[working["is_misclassified"]]
    low_confidence = working[working["confidence"] < 0.6]
    long_docs = working[working["text_len"] > working["text_len"].quantile(0.95)]

    misclassified.to_csv(output_dir / "misclassified.csv", index=False)
    low_confidence.to_csv(output_dir / "low_confidence.csv", index=False)
    long_docs.to_csv(output_dir / "long_documents.csv", index=False)

    confusion = (
        working.groupby(["y_true", "y_pred"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    confusion.to_csv(output_dir / "class_confusion.csv", index=False)

    bias_indicator = working.groupby("y_true")["is_misclassified"].mean().to_dict()
    summary = {
        "samples": int(len(working)),
        "misclassified": int(misclassified.shape[0]),
        "low_confidence": int(low_confidence.shape[0]),
        "long_documents": int(long_docs.shape[0]),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "bias_proxy.json", {str(k): float(v) for k, v in bias_indicator.items()})
    return summary
