"""Dataset loading and split management using Hugging Face Datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import DatasetDict, load_dataset

from .preprocessing import PreprocessConfig, preprocess_text


@dataclass(slots=True)
class DatasetSpec:
    hf_id: str
    text_column: str
    label_column: str
    subset: str | None = None


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "setfit_20_newsgroups": DatasetSpec(
        hf_id="SetFit/20_newsgroups",
        subset=None,
        text_column="text",
        label_column="label",
    ),
    "ag_news": DatasetSpec(hf_id="ag_news", subset=None, text_column="text", label_column="label"),
    "imdb": DatasetSpec(hf_id="imdb", subset=None, text_column="text", label_column="label"),
}


class DatasetLoader:
    """Load and standardize supported text-classification datasets."""

    def __init__(self, seed: int = 42, val_size: float = 0.1) -> None:
        self.seed = seed
        self.val_size = val_size

    def _rename_columns(self, dataset_dict: DatasetDict, spec: DatasetSpec) -> DatasetDict:
        for split, ds in dataset_dict.items():
            if spec.text_column != "text":
                ds = ds.rename_column(spec.text_column, "text")
            if spec.label_column != "label":
                ds = ds.rename_column(spec.label_column, "label")
            keep_cols = ["text", "label"]
            drop_cols = [col for col in ds.column_names if col not in keep_cols]
            dataset_dict[split] = ds.remove_columns(drop_cols)
        return dataset_dict

    def _ensure_validation_split(self, dataset_dict: DatasetDict) -> DatasetDict:
        if "validation" in dataset_dict:
            return dataset_dict

        train_valid = dataset_dict["train"].train_test_split(test_size=self.val_size, seed=self.seed)
        dataset_dict = DatasetDict(
            {
                "train": train_valid["train"],
                "validation": train_valid["test"],
                "test": dataset_dict.get("test", train_valid["test"]),
            }
        )
        return dataset_dict

    def load(self, name: str, preprocess_config: PreprocessConfig | None = None) -> DatasetDict:
        """Load dataset and standardize to text/label columns."""
        if name not in DATASET_REGISTRY:
            raise ValueError(f"Unsupported dataset: {name}")

        spec = DATASET_REGISTRY[name]
        dataset_dict = load_dataset(spec.hf_id, spec.subset) if spec.subset else load_dataset(spec.hf_id)
        dataset_dict = self._rename_columns(dataset_dict, spec)
        dataset_dict = self._ensure_validation_split(dataset_dict)

        if preprocess_config is not None:
            dataset_dict = dataset_dict.map(
                lambda row: {"text": preprocess_text(row["text"], preprocess_config)},
                desc=f"Preprocessing {name}",
            )

        return dataset_dict

    @staticmethod
    def label_names(dataset_dict: DatasetDict) -> list[str]:
        feature = dataset_dict["train"].features["label"]
        if getattr(feature, "names", None):
            return list(feature.names)

        labels = sorted(set(dataset_dict["train"]["label"]))
        return [str(label) for label in labels]

    @staticmethod
    def split_sizes(dataset_dict: DatasetDict) -> dict[str, int]:
        return {split: len(ds) for split, ds in dataset_dict.items()}

    @staticmethod
    def class_distribution(dataset_dict: DatasetDict) -> dict[int, int]:
        labels = dataset_dict["train"]["label"]
        counts: dict[int, int] = {}
        for label in labels:
            counts[int(label)] = counts.get(int(label), 0) + 1
        return counts

    @staticmethod
    def to_pandas(dataset_dict: DatasetDict) -> dict[str, Any]:
        return {split: ds.to_pandas() for split, ds in dataset_dict.items()}
