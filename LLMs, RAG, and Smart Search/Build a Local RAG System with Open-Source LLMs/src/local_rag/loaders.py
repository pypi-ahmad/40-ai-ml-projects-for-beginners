"""Document loaders for PDF, TXT, and Markdown files."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from local_rag.types import LoadedDocument
from local_rag.utils import sha256_text

MARKDOWN_SUFFIXES = {".md", ".markdown"}
TEXT_SUFFIXES = {".txt", ".rst", ".log", ".text", ".man"}
PDF_SUFFIXES = {".pdf"}
MANPAGE_SUFFIX_PATTERN = re.compile(r"^\.[1-9][a-z]*$")
DuplicatePolicy = Literal["skip_exact", "keep_all"]


class DocumentLoader:
    """Load documents from disk and normalize metadata."""

    def __init__(self, base_dir: Path, duplicate_policy: DuplicatePolicy = "skip_exact") -> None:
        self.base_dir = base_dir
        self.duplicate_policy = duplicate_policy

    def load_directory(self, path: Path | None = None) -> list[LoadedDocument]:
        """Load supported files recursively from directory."""

        root = path or self.base_dir
        documents: list[LoadedDocument] = []
        for file_path in sorted(self._iter_supported_files(root)):
            documents.extend(self.load_file(file_path))
        if self.duplicate_policy == "keep_all":
            return documents
        return self._deduplicate(documents)

    def load_file(self, file_path: Path) -> list[LoadedDocument]:
        """Load one file based on suffix."""

        suffix = file_path.suffix.lower()
        if suffix in PDF_SUFFIXES:
            return self._load_pdf(file_path)
        if suffix in MARKDOWN_SUFFIXES:
            return self._load_markdown(file_path)
        if suffix in TEXT_SUFFIXES or MANPAGE_SUFFIX_PATTERN.match(suffix):
            return self._load_text(file_path)
        return []

    def _iter_supported_files(self, root: Path) -> Iterator[Path]:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if (
                suffix in PDF_SUFFIXES | MARKDOWN_SUFFIXES | TEXT_SUFFIXES
                or MANPAGE_SUFFIX_PATTERN.match(suffix)
            ):
                yield path

    def _deduplicate(self, docs: list[LoadedDocument]) -> list[LoadedDocument]:
        seen_keys: set[tuple[str, str]] = set()
        deduped: list[LoadedDocument] = []
        for doc in docs:
            content_hash = str(doc.metadata.get("hash", ""))
            source_path = str(doc.metadata.get("source_path", ""))
            dedupe_key = (content_hash, source_path)
            if content_hash and dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            deduped.append(doc)
        return deduped

    def _relative_source_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.base_dir))
        except ValueError:
            return str(path)

    @staticmethod
    def _common_metadata(
        path: Path,
        source_rel: str,
        source_type: str,
        text: str,
    ) -> dict[str, object]:
        return {
            "source_path": source_rel,
            "source_type": source_type,
            "title": path.stem,
            "section": path.parent.name,
            "file_suffix": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
            "last_modified_ns": path.stat().st_mtime_ns,
            "hash": sha256_text(text),
        }

    def _load_pdf(self, path: Path) -> list[LoadedDocument]:
        try:
            reader = PdfReader(str(path))
        except PdfReadError:
            return []
        except Exception:  # noqa: BLE001
            return []
        records: list[LoadedDocument] = []
        source_rel = self._relative_source_path(path)

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:  # noqa: BLE001
                continue
            text = self._clean_text(text)
            if not text.strip():
                continue

            doc_id = sha256_text(f"{source_rel}::{page_number}")
            metadata = self._common_metadata(path, source_rel, "pdf", text) | {
                "doc_id": doc_id,
                "page_number": page_number,
            }
            records.append(LoadedDocument(doc_id=doc_id, text=text, metadata=metadata))

        return records

    def _load_text(self, path: Path) -> list[LoadedDocument]:
        text = path.read_text(encoding="utf-8", errors="replace")
        text = self._clean_text(text)
        if not text.strip():
            return []

        source_rel = self._relative_source_path(path)
        doc_id = sha256_text(source_rel)
        metadata = self._common_metadata(path, source_rel, "txt", text) | {
            "doc_id": doc_id,
            "page_number": None,
        }
        return [LoadedDocument(doc_id=doc_id, text=text, metadata=metadata)]

    def _load_markdown(self, path: Path) -> list[LoadedDocument]:
        text = path.read_text(encoding="utf-8", errors="replace")
        text = self._clean_markdown(text)
        if not text.strip():
            return []

        source_rel = self._relative_source_path(path)
        doc_id = sha256_text(source_rel)
        metadata = self._common_metadata(path, source_rel, "markdown", text) | {
            "doc_id": doc_id,
            "page_number": None,
        }
        return [LoadedDocument(doc_id=doc_id, text=text, metadata=metadata)]

    @staticmethod
    def _clean_markdown(text: str) -> str:
        text = re.sub(r"```[\s\S]*?```", " ", text)
        text = re.sub(r"`([^`]*)`", r"\1", text)
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
        text = text.replace("#", " ")
        return DocumentLoader._clean_text(text)

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.encode("utf-8", errors="replace").decode("utf-8")
        text = text.replace("\x00", " ")
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[\t ]{2,}", " ", text)
        return text.strip()
