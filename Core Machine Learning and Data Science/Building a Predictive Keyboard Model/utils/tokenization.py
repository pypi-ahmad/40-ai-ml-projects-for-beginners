"""Tokenizer backends and normalization pipeline."""

from __future__ import annotations

import re
import time
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import median
from typing import Iterable


def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_accents: bool = True,
    collapse_spaces: bool = True,
) -> str:
    """Normalize text before tokenization.

    Args:
        text: Raw input string.
        lowercase: Convert text to lowercase.
        strip_accents: Remove accent marks via NFKD decomposition.
        collapse_spaces: Collapse repeated whitespace into single spaces.

    Returns:
        Normalized text.
    """

    out = text
    if strip_accents:
        out = unicodedata.normalize("NFKD", out)
        out = "".join(ch for ch in out if not unicodedata.combining(ch))
    if lowercase:
        out = out.lower()
    if collapse_spaces:
        out = re.sub(r"\s+", " ", out).strip()
    return out


class TokenizerBackend(ABC):
    """Abstract tokenizer backend."""

    name: str

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Tokenize input text into string tokens."""


@dataclass(slots=True)
class RegexTokenizerBackend(TokenizerBackend):
    """Regex tokenizer baseline used as deterministic fallback."""

    name: str = "regex"
    pattern: re.Pattern[str] = re.compile(r"\b[a-zA-Z0-9']+\b")

    def tokenize(self, text: str) -> list[str]:
        return self.pattern.findall(text)


@dataclass(slots=True)
class NLTKTokenizerBackend(TokenizerBackend):
    """NLTK tokenizer backend using `word_tokenize`."""

    name: str = "nltk"

    def _ensure_resources(self) -> None:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            # punkt_tab exists on newer NLTK versions.
            nltk.download("punkt_tab", quiet=True)

    def tokenize(self, text: str) -> list[str]:
        try:
            self._ensure_resources()
            import nltk

            return [tok for tok in nltk.word_tokenize(text) if tok.strip()]
        except Exception:
            # Offline or missing-resource fallback.
            return RegexTokenizerBackend().tokenize(text)


@dataclass(slots=True)
class SpacyTokenizerBackend(TokenizerBackend):
    """spaCy tokenizer backend using lightweight blank English pipeline."""

    name: str = "spacy"

    def tokenize(self, text: str) -> list[str]:
        import spacy

        nlp = spacy.blank("en")
        return [token.text for token in nlp.make_doc(text) if token.text.strip()]


@dataclass(slots=True)
class HuggingFaceBPETokenizerBackend(TokenizerBackend):
    """Hugging Face Tokenizers BPE backend.

    Uses train-from-iterator approach based on official `tokenizers` docs.
    """

    vocab_size: int = 8_000
    min_frequency: int = 2
    name: str = "hf_bpe"

    def __post_init__(self) -> None:
        self._tokenizer = None

    def fit(self, corpus: Iterable[str]) -> None:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        )
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        self._tokenizer = tokenizer

    def tokenize(self, text: str) -> list[str]:
        if self._tokenizer is None:
            self.fit([text])
        assert self._tokenizer is not None
        encoded = self._tokenizer.encode(text)
        return encoded.tokens


def compare_tokenizer_outputs(
    text: str,
    backends: list[TokenizerBackend],
    *,
    normalize: bool = True,
    runs: int = 3,
) -> list[dict[str, float | int | str]]:
    """Compare tokenizers by token count and throughput.

    Returns one dictionary per backend for easy DataFrame conversion.
    """

    source = normalize_text(text) if normalize else text
    results: list[dict[str, float | int | str]] = []

    for backend in backends:
        # Exclude tokenizer fitting time from throughput comparisons.
        if hasattr(backend, "fit") and hasattr(backend, "_tokenizer"):
            if getattr(backend, "_tokenizer", None) is None:
                getattr(backend, "fit")([source])

        measured_tokens: list[str] = []
        elapsed_runs: list[float] = []
        for _ in range(max(1, runs)):
            start = time.perf_counter()
            measured_tokens = backend.tokenize(source)
            elapsed_runs.append(time.perf_counter() - start)

        elapsed = median(elapsed_runs)
        throughput = len(measured_tokens) / elapsed if elapsed > 0 else 0.0

        results.append(
            {
                "backend": backend.name,
                "token_count": len(measured_tokens),
                "unique_tokens": len(set(measured_tokens)),
                "throughput_tokens_per_sec": throughput,
                "sample_tokens": " ".join(measured_tokens[:15]),
            }
        )

    return results
