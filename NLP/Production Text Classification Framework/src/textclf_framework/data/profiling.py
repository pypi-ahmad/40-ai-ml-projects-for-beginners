"""Dataset profiling utilities."""

from __future__ import annotations

from collections import Counter

import numpy as np
from datasets import DatasetDict

from .preprocessing import token_statistics
from .schemas import DatasetProfile, LabelStats


def _outlier_count(lengths: list[int]) -> int:
    if not lengths:
        return 0
    q1 = np.percentile(lengths, 25)
    q3 = np.percentile(lengths, 75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    return int(sum(length > upper_bound for length in lengths))


def build_dataset_profile(dataset_name: str, dataset_dict: DatasetDict) -> DatasetProfile:
    """Build profiling summary for an already-loaded dataset."""
    train_texts = list(dataset_dict["train"]["text"])
    train_labels = list(dataset_dict["train"]["label"])
    label_counter = Counter(train_labels)

    lengths = [len(text) for text in train_texts]
    missing_count = sum(1 for text in train_texts if not str(text).strip())
    duplicate_count = len(train_texts) - len(set(train_texts))
    outliers = _outlier_count(lengths)

    label_names = getattr(dataset_dict["train"].features["label"], "names", None)
    total = len(train_labels)
    class_distribution: list[LabelStats] = []
    for label_id, count in sorted(label_counter.items()):
        label_name = label_names[label_id] if label_names and label_id < len(label_names) else str(label_id)
        class_distribution.append(
            LabelStats(
                label_id=int(label_id),
                label_name=label_name,
                count=int(count),
                proportion=float(count / max(total, 1)),
            )
        )

    token_stats = token_statistics(train_texts)
    vocab = set(" ".join(train_texts).split()) if train_texts else set()

    return DatasetProfile(
        dataset_name=dataset_name,
        split_sizes={split: len(ds) for split, ds in dataset_dict.items()},
        num_classes=len(class_distribution),
        class_distribution=class_distribution,
        missing_text_count=missing_count,
        duplicate_text_count=duplicate_count,
        outlier_count=outliers,
        avg_char_length=float(np.mean(lengths)) if lengths else 0.0,
        median_char_length=float(np.median(lengths)) if lengths else 0.0,
        p95_char_length=float(np.percentile(lengths, 95)) if lengths else 0.0,
        token_mean=token_stats["mean_tokens"],
        token_p95=token_stats["p95_tokens"],
        vocabulary_size=len(vocab),
    )
