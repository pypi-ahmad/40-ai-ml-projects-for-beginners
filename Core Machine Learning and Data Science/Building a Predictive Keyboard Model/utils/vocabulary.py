"""Vocabulary encoding/decoding helpers for language modeling."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

SPECIAL_TOKENS = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}


@dataclass(slots=True)
class Vocabulary:
    """Vocabulary with frequency-aware pruning.

    Args:
        min_freq: Keep only tokens seen at least this many times.
        max_size: Maximum size including special tokens.
    """

    min_freq: int = 1
    max_size: int | None = None
    word2idx: dict[str, int] = field(default_factory=dict, init=False)
    idx2word: dict[int, str] = field(default_factory=dict, init=False)
    token_frequencies: Counter[str] = field(default_factory=Counter, init=False)

    def __post_init__(self) -> None:
        self.word2idx = dict(SPECIAL_TOKENS)
        self.idx2word = {v: k for k, v in SPECIAL_TOKENS.items()}
        self.token_frequencies = Counter()

    def __len__(self) -> int:
        return len(self.word2idx)

    @property
    def pad_idx(self) -> int:
        return SPECIAL_TOKENS["<pad>"]

    @property
    def unk_idx(self) -> int:
        return SPECIAL_TOKENS["<unk>"]

    @property
    def bos_idx(self) -> int:
        return SPECIAL_TOKENS["<bos>"]

    @property
    def eos_idx(self) -> int:
        return SPECIAL_TOKENS["<eos>"]

    def build(self, tokenized_texts: Iterable[list[str]]) -> None:
        """Build token-to-index mappings from tokenized corpus."""

        self.token_frequencies = Counter()
        for tokens in tokenized_texts:
            self.token_frequencies.update(tokens)

        candidates = [
            (token, freq)
            for token, freq in self.token_frequencies.items()
            if freq >= self.min_freq
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))

        if self.max_size is not None:
            available = max(0, self.max_size - len(SPECIAL_TOKENS))
            candidates = candidates[:available]

        self.word2idx = dict(SPECIAL_TOKENS)
        self.idx2word = {v: k for k, v in SPECIAL_TOKENS.items()}
        for token, _ in candidates:
            if token in self.word2idx:
                continue
            idx = len(self.word2idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.word2idx.get(token, self.unk_idx) for token in tokens]

    def encode_with_special(
        self,
        tokens: list[str],
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_idx)
        ids.extend(self.encode(tokens))
        if add_eos:
            ids.append(self.eos_idx)
        return ids

    def decode(self, ids: list[int], remove_special: bool = True) -> list[str]:
        out: list[str] = []
        for token_id in ids:
            token = self.idx2word.get(token_id, "<unk>")
            if remove_special and token in SPECIAL_TOKENS:
                continue
            out.append(token)
        return out

    def statistics(self, tokens: list[str]) -> dict[str, float | int]:
        """Return vocabulary and OOV statistics for token list."""

        if not tokens:
            return {"vocab_size": len(self), "oov_rate": 0.0}

        oov_count = sum(1 for token in tokens if token not in self.word2idx)
        return {
            "vocab_size": len(self),
            "num_special_tokens": len(SPECIAL_TOKENS),
            "oov_rate": oov_count / len(tokens),
            "most_common_frequency": max(self.token_frequencies.values(), default=0),
            "rarest_frequency": min(self.token_frequencies.values(), default=0),
        }

    def save(self, path: Path) -> None:
        """Persist vocabulary mappings and frequencies."""

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "min_freq": self.min_freq,
            "max_size": self.max_size,
            "word2idx": self.word2idx,
            "token_frequencies": dict(self.token_frequencies),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        """Load vocabulary from JSON file."""

        data = json.loads(path.read_text(encoding="utf-8"))
        vocab = cls(
            min_freq=int(data.get("min_freq", 1)),
            max_size=data.get("max_size"),
        )
        vocab.word2idx = {k: int(v) for k, v in data["word2idx"].items()}
        vocab.idx2word = {idx: token for token, idx in vocab.word2idx.items()}
        vocab.token_frequencies = Counter(
            {k: int(v) for k, v in data.get("token_frequencies", {}).items()}
        )
        return vocab
