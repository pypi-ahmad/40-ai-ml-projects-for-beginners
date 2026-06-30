"""Sparse count-based n-gram language models."""

from __future__ import annotations

from collections import Counter, defaultdict

import torch
import torch.nn as nn


class _BaseSparseNgram(nn.Module):
    """Common sparse n-gram logic.

    The model stores context -> next-token counts in nested dictionaries,
    avoiding dense vocab^n tensors.
    """

    def __init__(self, vocab_size: int, history_len: int, smoothing: float = 1.0):
        super().__init__()
        if history_len < 0:
            raise ValueError("history_len must be >= 0")
        self.vocab_size = vocab_size
        self.history_len = history_len
        self.smoothing = smoothing
        self.context_counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        self.unigram_counts: Counter[int] = Counter()
        self.fitted = False

    def fit(self, corpus_ids: list[int]) -> None:
        self.context_counts = defaultdict(Counter)
        self.unigram_counts = Counter(corpus_ids)

        if self.history_len == 0:
            self.fitted = True
            return

        for idx in range(self.history_len, len(corpus_ids)):
            history = tuple(corpus_ids[idx - self.history_len : idx])
            token = corpus_ids[idx]
            self.context_counts[history][token] += 1

        self.fitted = True

    def _distribution_for_history(self, history: tuple[int, ...]) -> torch.Tensor:
        counts = self.context_counts.get(history)

        if counts is None or len(counts) == 0:
            # Backoff to smoothed unigram.
            total = sum(self.unigram_counts.values()) + self.smoothing * self.vocab_size
            probs = torch.full((self.vocab_size,), self.smoothing / total, dtype=torch.float32)
            for token, count in self.unigram_counts.items():
                probs[token] = (count + self.smoothing) / total
            return probs

        total = sum(counts.values()) + self.smoothing * self.vocab_size
        probs = torch.full((self.vocab_size,), self.smoothing / total, dtype=torch.float32)
        for token, count in counts.items():
            probs[token] = (count + self.smoothing) / total
        return probs

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Call fit() before forward().")

        if context.dim() != 2:
            raise ValueError("context must be 2D tensor [batch, seq]")

        batch_probs = []
        for row in context:
            if self.history_len == 0:
                history = tuple()
            else:
                history = tuple(row[-self.history_len :].tolist())
            probs = self._distribution_for_history(history).to(context.device)
            batch_probs.append(probs)

        stacked = torch.stack(batch_probs, dim=0)
        return torch.log(stacked + 1e-12)


class UnigramModel(_BaseSparseNgram):
    def __init__(self, vocab_size: int, smoothing: float = 1.0):
        super().__init__(vocab_size=vocab_size, history_len=0, smoothing=smoothing)


class MostFrequentWordModel(_BaseSparseNgram):
    """Most frequent token baseline.

    Always assigns highest probability to corpus most frequent token.
    """

    def __init__(self, vocab_size: int):
        super().__init__(vocab_size=vocab_size, history_len=0, smoothing=0.0)

    def _distribution_for_history(self, history: tuple[int, ...]) -> torch.Tensor:
        probs = torch.zeros(self.vocab_size, dtype=torch.float32)
        if not self.unigram_counts:
            probs[0] = 1.0
            return probs
        token, _ = self.unigram_counts.most_common(1)[0]
        probs[token] = 1.0
        return probs


class BigramModel(_BaseSparseNgram):
    def __init__(self, vocab_size: int, smoothing: float = 1.0):
        super().__init__(vocab_size=vocab_size, history_len=1, smoothing=smoothing)


class TrigramModel(_BaseSparseNgram):
    def __init__(self, vocab_size: int, smoothing: float = 1.0):
        super().__init__(vocab_size=vocab_size, history_len=2, smoothing=smoothing)


class NgramModel(_BaseSparseNgram):
    """Generic n-gram model where `n` is order (1=unigram,2=bigram,...)."""

    def __init__(self, n: int, vocab_size: int, smoothing: float = 1.0):
        if n < 1:
            raise ValueError("n must be >= 1")
        super().__init__(vocab_size=vocab_size, history_len=n - 1, smoothing=smoothing)
        self.n = n
