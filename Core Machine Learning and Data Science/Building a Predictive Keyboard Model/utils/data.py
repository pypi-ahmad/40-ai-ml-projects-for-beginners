"""Dataset ingestion utilities for predictive keyboard project."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .config import PathConfig

try:
    from datasets import get_dataset_split_names, load_dataset
except ImportError:  # pragma: no cover - handled at runtime when dependency missing
    get_dataset_split_names = None
    load_dataset = None

WORD_PATTERN = re.compile(r"\b[a-zA-Z0-9']+\b")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n")


@dataclass(slots=True)
class CorpusBundle:
    """Train/val/test text bundles with source metadata."""

    train_text: str
    val_text: str
    test_text: str
    metadata: dict[str, str | int | float]


def extract_zip_dataset(zip_path: Path, extract_to: Path) -> list[Path]:
    """Extract zip archive and return extracted file paths.

    Args:
        zip_path: Path to zip file.
        extract_to: Output directory.

    Returns:
        List of extracted file paths.
    """

    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(extract_to)
        names = zip_file.namelist()

    extracted = []
    for name in names:
        candidate = extract_to / name
        if candidate.exists() and candidate.is_file() and "__MACOSX" not in name:
            extracted.append(candidate)
    return extracted


def read_text_file(path: Path) -> str:
    """Read UTF-8 text with fallback error replacement."""

    return path.read_text(encoding="utf-8", errors="replace")


def corpus_statistics(text: str) -> dict[str, int | float]:
    """Compute corpus profile statistics used in notebooks and README."""

    words = WORD_PATTERN.findall(text.lower())
    sentences = [s for s in SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]
    documents = [p for p in PARAGRAPH_SPLIT_PATTERN.split(text) if p.strip()]

    return {
        "num_documents": len(documents),
        "num_sentences": len(sentences),
        "num_words": len(words),
        "num_unique_tokens": len(set(words)),
        "avg_sentence_length": (len(words) / len(sentences)) if sentences else 0.0,
    }


def _slice_by_token_budget(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    word_matches = list(WORD_PATTERN.finditer(text))
    if len(word_matches) <= token_budget:
        return text
    end_char = word_matches[token_budget - 1].end()
    # Keep deterministic text prefix with punctuation and structure.
    return text[:end_char].strip()


def _split_text_train_val_test(text: str) -> tuple[str, str, str]:
    word_matches = list(WORD_PATTERN.finditer(text))
    if not word_matches:
        return text.strip(), "", ""

    n_words = len(word_matches)
    train_end_word = max(int(0.8 * n_words), 1)
    val_end_word = max(int(0.9 * n_words), train_end_word)

    train_end_char = word_matches[train_end_word - 1].end()
    val_end_char = word_matches[val_end_word - 1].end()

    train = text[:train_end_char].strip()
    val = text[train_end_char:val_end_char].strip()
    test = text[val_end_char:].strip()
    return train, val, test


def validate_text_corpus(text: str) -> dict[str, int | float]:
    """Validate one corpus split for integrity and quality diagnostics."""

    raw_docs = PARAGRAPH_SPLIT_PATTERN.split(text)
    docs = [doc for doc in raw_docs if doc.strip()]
    normalized_docs = [re.sub(r"\s+", " ", doc.strip().lower()) for doc in docs]
    duplicate_doc_count = len(normalized_docs) - len(set(normalized_docs))
    empty_docs = sum(1 for doc in raw_docs if not doc.strip())

    words = WORD_PATTERN.findall(text.lower())
    sentences = [s for s in SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]

    return {
        "num_documents": len(docs),
        "num_sentences": len(sentences),
        "num_words": len(words),
        "num_unique_tokens": len(set(words)),
        "duplicate_documents": duplicate_doc_count,
        "empty_documents": empty_docs,
        "replacement_characters": text.count("\ufffd"),
        "avg_document_length_words": (len(words) / len(docs)) if docs else 0.0,
    }


def validate_corpus_bundle(bundle: CorpusBundle) -> dict[str, dict[str, int | float]]:
    """Run corpus integrity checks across train/val/test splits."""

    report = {
        "train": validate_text_corpus(bundle.train_text),
        "val": validate_text_corpus(bundle.val_text),
        "test": validate_text_corpus(bundle.test_text),
    }
    report["combined"] = validate_text_corpus(
        f"{bundle.train_text}\n\n{bundle.val_text}\n\n{bundle.test_text}"
    )
    return report


def download_wikitext_subset(
    data_dir: Path,
    max_train_tokens: int,
    max_val_tokens: int,
    max_test_tokens: int,
) -> CorpusBundle:
    """Download WikiText-103 and build deterministic token-limited subset.

    Uses Hugging Face `datasets` package and caches data under project `data` folder.
    """

    if load_dataset is None or get_dataset_split_names is None:
        raise ImportError("datasets package not installed. Run `uv sync` first.")

    cache_dir = data_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = "Salesforce/wikitext"
    split_names = get_dataset_split_names(dataset_id, "wikitext-103-v1")
    required = {"train", "validation", "test"}
    if not required.issubset(set(split_names)):
        raise ValueError(
            f"WikiText split mismatch. Expected {required}, got {set(split_names)}"
        )

    dataset = load_dataset(dataset_id, "wikitext-103-v1", cache_dir=str(cache_dir))

    def split_to_text(split_name: str, budget: int) -> str:
        lines = [line for line in dataset[split_name]["text"] if line and line.strip()]
        text = "\n".join(lines)
        return _slice_by_token_budget(text, budget)

    train_text = split_to_text("train", max_train_tokens)
    val_text = split_to_text("validation", max_val_tokens)
    test_text = split_to_text("test", max_test_tokens)

    return CorpusBundle(
        train_text=train_text,
        val_text=val_text,
        test_text=test_text,
        metadata={
            "source": f"{dataset_id}:wikitext-103-v1",
            "train_token_budget": max_train_tokens,
            "val_token_budget": max_val_tokens,
            "test_token_budget": max_test_tokens,
        },
    )


def prepare_combined_corpus(
    project_root: Path,
    include_wikitext: bool = True,
    wikitext_train_tokens: int = 1_500_000,
    wikitext_val_tokens: int = 200_000,
    wikitext_test_tokens: int = 200_000,
) -> CorpusBundle:
    """Create combined Sherlock + WikiText corpus with train/val/test splits.

    Args:
        project_root: Project root directory.
        include_wikitext: Include external corpus if True.
        wikitext_train_tokens: Token budget for WikiText train subset.
        wikitext_val_tokens: Token budget for WikiText validation subset.
        wikitext_test_tokens: Token budget for WikiText test subset.

    Returns:
        CorpusBundle with merged splits.

    Example:
        >>> bundle = prepare_combined_corpus(Path.cwd())
        >>> len(bundle.train_text) > 0
        True
    """

    paths = PathConfig.from_project_root(project_root)
    paths.ensure_dirs()

    zip_path = project_root / "3dd01-book.zip"
    extract_dir = paths.data_dir / "raw"
    extracted_files = extract_zip_dataset(zip_path, extract_dir)

    sherlock_candidates = [
        p for p in extracted_files if p.suffix.lower() == ".txt" and "MACOSX" not in str(p)
    ]
    if not sherlock_candidates:
        fallback = project_root / "sherlock-holm.es_stories_plain-text_advs.txt"
        if fallback.exists():
            sherlock_candidates = [fallback]

    if not sherlock_candidates:
        raise FileNotFoundError("Could not find extracted Sherlock text file.")

    sherlock_text = read_text_file(sherlock_candidates[0])
    s_train, s_val, s_test = _split_text_train_val_test(sherlock_text)

    metadata: dict[str, str | int | float] = {
        "primary_source": sherlock_candidates[0].name,
        "include_wikitext": int(include_wikitext),
    }

    if include_wikitext:
        try:
            wiki = download_wikitext_subset(
                data_dir=paths.data_dir,
                max_train_tokens=wikitext_train_tokens,
                max_val_tokens=wikitext_val_tokens,
                max_test_tokens=wikitext_test_tokens,
            )
            train_text = f"{s_train}\n\n{wiki.train_text}".strip()
            val_text = f"{s_val}\n\n{wiki.val_text}".strip()
            test_text = f"{s_test}\n\n{wiki.test_text}".strip()
            metadata.update(wiki.metadata)
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            train_text, val_text, test_text = s_train, s_val, s_test
            metadata.update(
                {
                    "wikitext_fallback": 1,
                    "wikitext_error": str(exc),
                }
            )
    else:
        train_text, val_text, test_text = s_train, s_val, s_test

    return CorpusBundle(
        train_text=train_text,
        val_text=val_text,
        test_text=test_text,
        metadata=metadata,
    )


def save_corpus_bundle(bundle: CorpusBundle, out_dir: Path) -> dict[str, Path]:
    """Persist corpus bundle and metadata to disk."""

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.txt"
    val_path = out_dir / "val.txt"
    test_path = out_dir / "test.txt"
    meta_path = out_dir / "metadata.json"

    train_path.write_text(bundle.train_text, encoding="utf-8")
    val_path.write_text(bundle.val_text, encoding="utf-8")
    test_path.write_text(bundle.test_text, encoding="utf-8")
    meta_path.write_text(json.dumps(bundle.metadata, indent=2), encoding="utf-8")

    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "metadata": meta_path,
    }
