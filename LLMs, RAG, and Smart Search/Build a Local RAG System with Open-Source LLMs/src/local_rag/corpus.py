"""Corpus profiling, validation, and quickstart sampling utilities."""

from __future__ import annotations

import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from local_rag.utils import json_dump

MARKDOWN_SUFFIXES = {".md", ".markdown"}
TEXT_SUFFIXES = {".txt", ".rst", ".log", ".text", ".man"}
PDF_SUFFIXES = {".pdf"}
MANPAGE_SUFFIX_PATTERN = re.compile(r"^\.[1-9][a-z]*$")


def _source_type(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return "pdf"
    if suffix in MARKDOWN_SUFFIXES:
        return "markdown"
    if suffix in TEXT_SUFFIXES or MANPAGE_SUFFIX_PATTERN.match(suffix):
        return "text"
    return None


@dataclass(slots=True)
class CorpusStats:
    """High-level dataset profile for reporting and validation."""

    documents_dir: str
    total_files: int
    pdf_files: int
    markdown_files: int
    text_files: int
    total_size_mb: float

    @property
    def has_mixed_formats(self) -> bool:
        """True when PDF/markdown/text all present."""

        return self.pdf_files > 0 and self.markdown_files > 0 and self.text_files > 0


@dataclass(slots=True)
class CorpusValidation:
    """Validation output for corpus quality gates."""

    ok: bool
    errors: list[str]
    stats: CorpusStats


def corpus_stats(documents_dir: Path) -> CorpusStats:
    """Compute supported format counts and total size."""

    pdf_files = 0
    markdown_files = 0
    text_files = 0
    total_size = 0
    total_files = 0

    for path in documents_dir.rglob("*"):
        if not path.is_file():
            continue
        source_type = _source_type(path)
        if source_type is None:
            continue
        total_files += 1
        total_size += path.stat().st_size
        if source_type == "pdf":
            pdf_files += 1
        elif source_type == "markdown":
            markdown_files += 1
        else:
            text_files += 1

    return CorpusStats(
        documents_dir=str(documents_dir),
        total_files=total_files,
        pdf_files=pdf_files,
        markdown_files=markdown_files,
        text_files=text_files,
        total_size_mb=total_size / (1024 * 1024),
    )


def validate_corpus(
    documents_dir: Path,
    *,
    min_total_files: int = 10,
    min_pdf_files: int = 1,
    min_markdown_files: int = 1,
    min_text_files: int = 1,
) -> CorpusValidation:
    """Validate corpus completeness for production RAG usage."""

    stats = corpus_stats(documents_dir)
    errors: list[str] = []

    if stats.total_files < min_total_files:
        errors.append(
            f"Insufficient corpus size: {stats.total_files} files < required {min_total_files}."
        )
    if stats.pdf_files < min_pdf_files:
        errors.append(
            f"Insufficient PDF files: {stats.pdf_files} < required {min_pdf_files}."
        )
    if stats.markdown_files < min_markdown_files:
        errors.append(
            f"Insufficient Markdown files: {stats.markdown_files} < required {min_markdown_files}."
        )
    if stats.text_files < min_text_files:
        errors.append(
            f"Insufficient text/man files: {stats.text_files} < required {min_text_files}."
        )

    return CorpusValidation(ok=not errors, errors=errors, stats=stats)


def write_corpus_manifest(path: Path, stats: CorpusStats) -> None:
    """Persist corpus profile report to JSON."""

    payload = asdict(stats) | {"has_mixed_formats": stats.has_mixed_formats}
    json_dump(path, payload)


def build_quickstart_corpus(
    *,
    source_dir: Path,
    target_dir: Path,
    max_pdf: int = 20,
    max_markdown: int = 40,
    max_text: int = 120,
) -> CorpusStats:
    """Create deterministic sampled mixed-format quickstart corpus."""

    validation = validate_corpus(
        source_dir,
        min_total_files=3,
        min_pdf_files=1,
        min_markdown_files=1,
        min_text_files=1,
    )
    if not validation.ok:
        error_msg = "; ".join(validation.errors)
        raise ValueError(
            "Cannot build quickstart corpus from invalid source corpus: "
            f"{error_msg}"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    for existing in target_dir.rglob("*"):
        if existing.is_file():
            existing.unlink()

    limits = {
        "pdf": max_pdf,
        "markdown": max_markdown,
        "text": max_text,
    }
    copied = {"pdf": 0, "markdown": 0, "text": 0}

    for source_path in sorted(source_dir.rglob("*")):
        if not source_path.is_file():
            continue
        source_type = _source_type(source_path)
        if source_type is None:
            continue
        if copied[source_type] >= limits[source_type]:
            continue

        prefix = {
            "pdf": "pdf",
            "markdown": "md",
            "text": "txt",
        }[source_type]
        index = copied[source_type] + 1
        destination_dir = target_dir / source_type
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{prefix}_{index:04d}{source_path.suffix.lower()}"
        shutil.copy2(source_path, destination)
        copied[source_type] += 1

        if all(copied[key] >= limits[key] for key in limits):
            break

    return corpus_stats(target_dir)
