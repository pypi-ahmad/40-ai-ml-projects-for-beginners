"""Document loaders for PDF, TXT, and Markdown files."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pypdf import PdfReader

from local_rag.types import LoadedDocument
from local_rag.utils import sha256_file, sha256_text

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
        seen_hashes: set[str] = set()
        deduped: list[LoadedDocument] = []
        for doc in docs:
            content_hash = str(doc.metadata.get("hash", ""))
            if content_hash and content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            deduped.append(doc)
        return deduped

    def _relative_source_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.base_dir))
        except ValueError:
            return str(path)

    @staticmethod
    def _domain_from_path(path: Path) -> str:
        """Infer document domain from top-level parent folder."""

        parent = path.parent.name.lower()
        if any(token in parent for token in ("policy", "legal", "compliance", "regulation")):
            return "policy"
        if any(token in parent for token in ("research", "paper", "arxiv", "journal")):
            return "research"
        if any(token in parent for token in ("finance", "annual", "report", "sec")):
            return "finance"
        return "technical"

    def _common_metadata(
        self,
        path: Path,
        source_rel: str,
        source_type: str,
        text: str,
    ) -> dict[str, object]:
        """Build common metadata shared by all loaders."""

        file_hash = sha256_file(path)
        document_id = sha256_text(source_rel)
        version_id = sha256_text(f"{file_hash}::{path.stat().st_mtime_ns}")
        return {
            "source_path": source_rel,
            "source_type": source_type,
            "title": path.stem,
            "document_name": path.name,
            "document_id": document_id,
            "domain": self._domain_from_path(path),
            "section": path.parent.name,
            "file_suffix": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
            "last_modified_ns": path.stat().st_mtime_ns,
            "file_hash": file_hash,
            "version_id": version_id,
            "hash": sha256_text(text),
        }

    def _load_pdf(self, path: Path) -> list[LoadedDocument]:
        reader = PdfReader(str(path))
        records: list[LoadedDocument] = []
        source_rel = self._relative_source_path(path)
        base_meta = self._common_metadata(path, source_rel, "pdf", text="")
        document_id = str(base_meta["document_id"])

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = self._clean_text(text)
            if not text.strip():
                continue

            doc_id = sha256_text(f"{document_id}::{page_number}")
            metadata = dict(base_meta)
            metadata["hash"] = sha256_text(text)
            metadata["doc_id"] = doc_id
            metadata["page_number"] = page_number
            records.append(LoadedDocument(doc_id=doc_id, text=text, metadata=metadata))

        return records

    def _load_text(self, path: Path) -> list[LoadedDocument]:
        text = path.read_text(encoding="utf-8", errors="replace")
        text = self._clean_text(text)
        if not text.strip():
            return []

        source_rel = self._relative_source_path(path)
        metadata = self._common_metadata(path, source_rel, "txt", text)
        metadata["doc_id"] = sha256_text(source_rel)
        metadata["page_number"] = None
        return [LoadedDocument(doc_id=str(metadata["doc_id"]), text=text, metadata=metadata)]

    def _load_markdown(self, path: Path) -> list[LoadedDocument]:
        text = path.read_text(encoding="utf-8", errors="replace")
        text = self._clean_markdown(text)
        if not text.strip():
            return []

        source_rel = self._relative_source_path(path)
        metadata = self._common_metadata(path, source_rel, "markdown", text)
        metadata["doc_id"] = sha256_text(source_rel)
        metadata["page_number"] = None
        return [LoadedDocument(doc_id=str(metadata["doc_id"]), text=text, metadata=metadata)]

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
