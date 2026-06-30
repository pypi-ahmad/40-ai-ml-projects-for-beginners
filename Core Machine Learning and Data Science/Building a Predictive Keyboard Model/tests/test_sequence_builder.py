import torch

from utils.sequence_builder import (
    LanguageModelDataset,
    build_context_target_pairs,
    build_dataloaders_from_ids,
)


def test_build_context_target_pairs() -> None:
    ids = [1, 2, 3, 4, 5]
    pairs = build_context_target_pairs(ids, context_len=3)
    assert pairs == [([1, 2, 3], 4), ([2, 3, 4], 5)]


def test_language_model_dataset_and_dataloaders() -> None:
    pairs = [([1, 2, 3], 4), ([2, 3, 4], 5), ([3, 4, 5], 6), ([4, 5, 6], 7)]
    dataset = LanguageModelDataset(pairs)
    ctx, tgt = dataset[0]

    assert isinstance(ctx, torch.Tensor)
    assert isinstance(tgt, torch.Tensor)
    assert ctx.shape == (3,)
    assert tgt.item() == 4

    loaders = build_dataloaders_from_ids(
        ids=[1, 2, 3, 4, 5, 6, 7],
        context_len=3,
        batch_size=2,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=7,
    )
    assert set(loaders.keys()) == {"train", "val", "test"}


def test_split_safe_dataloaders_do_not_cross_token_boundaries() -> None:
    ids = list(range(1, 31))
    loaders = build_dataloaders_from_ids(
        ids=ids,
        context_len=3,
        batch_size=4,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        split_mode="contiguous",
    )

    train_dataset = loaders["train"].dataset
    val_dataset = loaders["val"].dataset
    test_dataset = loaders["test"].dataset

    # Train token split ends before id 19 for this configuration.
    assert all(pair[1] < 19 for pair in train_dataset.pairs)
    # Validation targets come from the middle token range.
    assert all(19 <= pair[1] <= 24 for pair in val_dataset.pairs)
    # Test targets come from the final token range.
    assert all(pair[1] >= 25 for pair in test_dataset.pairs)
