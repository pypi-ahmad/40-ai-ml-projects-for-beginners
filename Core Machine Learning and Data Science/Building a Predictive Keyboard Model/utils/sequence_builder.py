"""Sequence construction utilities for next-word prediction."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset


def build_context_target_pairs(ids: list[int], context_len: int) -> list[tuple[list[int], int]]:
    """Convert token IDs into fixed-window context-target pairs."""

    if context_len < 1:
        raise ValueError("context_len must be >= 1")

    return [
        (ids[idx - context_len : idx], ids[idx])
        for idx in range(context_len, len(ids))
    ]


class LanguageModelDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch dataset for next-token language modeling."""

    def __init__(self, pairs: list[tuple[list[int], int]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        context, target = self.pairs[index]
        return torch.tensor(context, dtype=torch.long), torch.tensor(
            target, dtype=torch.long
        )


@dataclass(slots=True)
class SequenceBuilder:
    """Helper class for building sequence datasets across context windows."""

    context_len: int = 5

    def build(self, ids: list[int]) -> list[tuple[list[int], int]]:
        return build_context_target_pairs(ids, self.context_len)


def build_dataloaders_from_ids(
    ids: list[int],
    *,
    context_len: int,
    batch_size: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    split_mode: Literal["contiguous", "shuffled"] = "contiguous",
    shuffle_train: bool = True,
) -> dict[str, DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    """Create train/val/test dataloaders from token IDs.

    Default behavior uses split-safe contiguous token boundaries, then builds
    context-target pairs inside each split to avoid cross-split leakage.
    """

    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must satisfy val+test < 1")
    if split_mode not in {"contiguous", "shuffled"}:
        raise ValueError("split_mode must be 'contiguous' or 'shuffled'")

    n_ids = len(ids)
    test_ids_count = int(n_ids * test_ratio)
    val_ids_count = int(n_ids * val_ratio)
    train_ids_count = n_ids - val_ids_count - test_ids_count

    train_ids = ids[:train_ids_count]
    val_ids = ids[train_ids_count : train_ids_count + val_ids_count]
    test_ids = ids[train_ids_count + val_ids_count :]

    train_pairs = build_context_target_pairs(train_ids, context_len)
    val_pairs = build_context_target_pairs(val_ids, context_len)
    test_pairs = build_context_target_pairs(test_ids, context_len)

    if split_mode == "shuffled":
        rng = random.Random(seed)
        rng.shuffle(train_pairs)
        rng.shuffle(val_pairs)
        rng.shuffle(test_pairs)

    train_dataset = LanguageModelDataset(train_pairs)
    val_dataset = LanguageModelDataset(val_pairs)
    test_dataset = LanguageModelDataset(test_pairs)

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }
