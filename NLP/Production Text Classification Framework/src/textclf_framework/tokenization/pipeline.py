"""Tokenization pipeline with dynamic padding and long-document support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerBase


@dataclass(slots=True)
class TokenizationConfig:
    model_name: str
    max_length: int = 256
    truncation: bool = True
    padding: bool | str = False
    stride: int = 64


class TokenizationPipeline:
    """Applies tokenizer transforms to datasets."""

    def __init__(self, config: TokenizationConfig) -> None:
        self.config = config
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    def tokenize_batch(self, examples: dict[str, list[str]]) -> dict[str, Any]:
        return self.tokenizer(
            examples["text"],
            truncation=self.config.truncation,
            max_length=self.config.max_length,
            padding=self.config.padding,
        )

    def tokenize_dataset(self, dataset_dict: DatasetDict) -> DatasetDict:
        tokenized = dataset_dict.map(self.tokenize_batch, batched=True, desc="Tokenizing dataset")
        return tokenized

    def collator(self) -> DataCollatorWithPadding:
        return DataCollatorWithPadding(tokenizer=self.tokenizer, padding="longest")

    def sliding_window_encode(self, text: str) -> list[dict[str, list[int]]]:
        encoded = self.tokenizer(
            text,
            truncation=False,
            return_overflowing_tokens=True,
            max_length=self.config.max_length,
            stride=self.config.stride,
        )
        windows: list[dict[str, list[int]]] = []
        for idx in range(len(encoded["input_ids"])):
            windows.append(
                {
                    "input_ids": encoded["input_ids"][idx],
                    "attention_mask": encoded["attention_mask"][idx],
                }
            )
        return windows

    def max_sequence_benchmark(self, dataset: Dataset, sample_size: int = 1000) -> dict[str, float]:
        texts = dataset["text"][: min(sample_size, len(dataset))]
        lengths = [len(self.tokenizer(t, truncation=False)["input_ids"]) for t in texts]
        if not lengths:
            return {"mean": 0.0, "p95": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(lengths)),
            "p95": float(np.percentile(lengths, 95)),
            "max": float(np.max(lengths)),
        }
