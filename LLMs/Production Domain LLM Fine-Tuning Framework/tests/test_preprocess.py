import pandas as pd
from datasets import Dataset

from domain_llm_ft.data.preprocess import deduplicate_dataset, normalize_text


def test_normalize_text() -> None:
    assert normalize_text("  hello   world  ") == "hello world"


def test_deduplicate_dataset() -> None:
    ds = Dataset.from_pandas(
        pd.DataFrame(
            {
                "text": ["a", "a", "b"],
                "label": [0, 0, 1],
            }
        ),
        preserve_index=False,
    )
    out = deduplicate_dataset(ds, "text")
    assert len(out) == 2
