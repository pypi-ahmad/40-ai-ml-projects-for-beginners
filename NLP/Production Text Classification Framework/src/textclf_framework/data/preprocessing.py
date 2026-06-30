"""Text preprocessing pipeline."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

import emoji
from langdetect import LangDetectException, detect

try:
    from nltk.stem import PorterStemmer, WordNetLemmatizer
except Exception:  # pragma: no cover - optional dependency behavior
    PorterStemmer = None
    WordNetLemmatizer = None


_HTML_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class PreprocessConfig:
    lowercase: bool = True
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    emoji_policy: str = "demojize"  # remove | demojize | keep
    stemming: bool = False
    lemmatization: bool = False


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _maybe_stem_or_lemmatize(text: str, config: PreprocessConfig) -> str:
    tokens = text.split()
    if config.stemming and PorterStemmer is not None:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(tok) for tok in tokens]
    if config.lemmatization and WordNetLemmatizer is not None:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    return " ".join(tokens)


def preprocess_text(text: str | None, config: PreprocessConfig | None = None) -> str:
    """Apply deterministic normalization and cleanup transforms."""
    cfg = config or PreprocessConfig()
    normalized = _normalize_unicode(text or "")
    normalized = html.unescape(normalized)

    if cfg.remove_html:
        normalized = _HTML_RE.sub(" ", normalized)
    if cfg.remove_urls:
        normalized = _URL_RE.sub(" ", normalized)
    if cfg.remove_emails:
        normalized = _EMAIL_RE.sub(" ", normalized)

    if cfg.emoji_policy == "remove":
        normalized = emoji.replace_emoji(normalized, replace="")
    elif cfg.emoji_policy == "demojize":
        normalized = emoji.demojize(normalized, delimiters=(" ", " "))

    if cfg.lowercase:
        normalized = normalized.lower()

    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    normalized = _maybe_stem_or_lemmatize(normalized, cfg)
    return normalized


def detect_language(text: str) -> str:
    """Detect language code safely."""
    if not text.strip():
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def token_statistics(texts: Iterable[str]) -> dict[str, float]:
    """Compute basic token length stats without model tokenizer."""
    lengths = [len(t.split()) for t in texts if t]
    if not lengths:
        return {"mean_tokens": 0.0, "max_tokens": 0.0, "p95_tokens": 0.0}

    lengths_sorted = sorted(lengths)
    p95_idx = int(0.95 * (len(lengths_sorted) - 1))
    return {
        "mean_tokens": float(sum(lengths) / len(lengths)),
        "max_tokens": float(max(lengths)),
        "p95_tokens": float(lengths_sorted[p95_idx]),
    }
