"""Tokenizer factory and preprocessing."""

from __future__ import annotations

from collections.abc import Mapping

from datasets import DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerBase

from domain_llm_ft.config.schemas import TokenizerConfig


class TokenizationPipeline:
    """Tokenization pipeline for HF datasets."""

    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.name, use_fast=True)

    def tokenize_dataset(
        self,
        dataset: DatasetDict,
        text_column: str,
        label_column: str,
    ) -> DatasetDict:
        """Tokenize full dataset dictionary."""

        def _tokenize(batch: Mapping[str, list]) -> dict[str, list]:
            return self.tokenizer(
                batch[text_column],
                max_length=self.config.max_length,
                truncation=self.config.truncation,
                padding=self.config.padding,
            )

        tokenized = dataset.map(_tokenize, batched=True)
        tokenized = tokenized.rename_column(label_column, "labels")
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return tokenized

    def build_collator(self) -> DataCollatorWithPadding:
        """Create dynamic padding collator."""
        return DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8 if self.config.dynamic_padding else None,
        )


def load_tokenizer(name: str) -> PreTrainedTokenizerBase:
    """Load tokenizer for standalone use."""
    return AutoTokenizer.from_pretrained(name, use_fast=True)
