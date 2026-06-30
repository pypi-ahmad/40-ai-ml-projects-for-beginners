"""Dataset loading from Hugging Face and local files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from peft_platform.data.schemas import Sample


def _normalize_row(row: dict[str, Any], task_type: str) -> Sample:
    instruction = str(row.get("instruction", ""))
    model_input = str(row.get("input", row.get("context", "")))
    output = str(row.get("output", row.get("response", row.get("text", ""))))
    label = row.get("label")
    messages = row.get("messages", [])

    if messages and isinstance(messages, list):
        normalized_messages = [
            {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
            for item in messages
            if isinstance(item, dict)
        ]
    else:
        normalized_messages = []

    return Sample(
        task_type=task_type,
        instruction=instruction,
        input=model_input,
        output=output,
        label=label,
        messages=normalized_messages,
        metadata={k: v for k, v in row.items() if k not in {"instruction", "input", "output", "label", "messages"}},
    )


def load_local_csv(path: Path, task_type: str) -> list[Sample]:
    import pandas as pd

    frame = pd.read_csv(path)
    return [_normalize_row(record, task_type) for record in frame.to_dict(orient="records")]


def load_local_jsonl(path: Path, task_type: str) -> list[Sample]:
    import pandas as pd

    frame = pd.read_json(path, lines=True)
    return [_normalize_row(record, task_type) for record in frame.to_dict(orient="records")]


def load_hf_dataset(name: str, split: str, task_type: str) -> list[Sample]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets package unavailable") from exc

    ds = load_dataset(name, split=split)
    return [_normalize_row(record, task_type) for record in ds]  # type: ignore[arg-type]
