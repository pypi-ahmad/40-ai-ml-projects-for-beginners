"""Document ingestion for local knowledge base files."""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from markdownify import markdownify
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from hybrid_research_assistant.schemas import DocumentRecord
from hybrid_research_assistant.utils import sha256_file, sha256_text

MARKDOWN_SUFFIXES = {".md", ".markdown"}
TEXT_SUFFIXES = {".txt", ".rst", ".log", ".text", ".csv"}
PDF_SUFFIXES = {".pdf"}
HTML_SUFFIXES = {".html", ".htm"}
DOCX_SUFFIXES = {".docx"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
DuplicatePolicy = Literal["skip_exact", "keep_all"]


class OCRBackend(Protocol):
    """OCR backend protocol."""

    def extract_text(self, file_path: Path) -> str:
        """Extract text from image-like files."""


class NullOCRBackend:
    """No-op OCR backend when OCR is not available."""

    def extract_text(self, file_path: Path) -> str:
        return ""


class DocumentLoader:
    """Load documents recursively and normalize metadata."""

    def __init__(
        self,
        base_dir: Path,
        *,
        namespace: str,
        duplicate_policy: DuplicatePolicy = "skip_exact",
        ocr_backend: OCRBackend | None = None,
    ) -> None:
        self.base_dir = base_dir
        self.namespace = namespace
        self.duplicate_policy = duplicate_policy
        self.ocr_backend = ocr_backend or NullOCRBackend()

    def load_directory(self, path: Path | None = None) -> list[DocumentRecord]:
        """Load all supported documents from a directory tree."""

        root = path or self.base_dir
        records: list[DocumentRecord] = []
        for file_path in sorted(self._iter_supported_files(root)):
            records.extend(self.load_file(file_path))
        if self.duplicate_policy == "keep_all":
            return records
        return self._deduplicate(records)

    def load_file(self, path: Path) -> list[DocumentRecord]:
        """Load one file and return normalized records."""

        suffix = path.suffix.lower()
        if suffix in PDF_SUFFIXES:
            return self._load_pdf(path)
        if suffix in MARKDOWN_SUFFIXES:
            return self._load_markdown(path)
        if suffix in TEXT_SUFFIXES:
            return self._load_text(path)
        if suffix in HTML_SUFFIXES:
            return self._load_html(path)
        if suffix in DOCX_SUFFIXES:
            return self._load_docx(path)
        if suffix in IMAGE_SUFFIXES:
            return self._load_image(path)
        return []

    def _iter_supported_files(self, root: Path) -> Iterator[Path]:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in (
                PDF_SUFFIXES
                | MARKDOWN_SUFFIXES
                | TEXT_SUFFIXES
                | HTML_SUFFIXES
                | DOCX_SUFFIXES
                | IMAGE_SUFFIXES
            ):
                yield path

    def _relative_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.base_dir))
        except ValueError:
            return str(path)

    def _base_metadata(self, path: Path, text: str, *, source_type: str) -> dict[str, object]:
        source = self._relative_path(path)
        doc_hash = sha256_file(path)
        now = datetime.now(UTC).isoformat()
        return {
            "source": source,
            "filename": path.name,
            "source_path": source,
            "source_type": source_type,
            "section": path.parent.name,
            "document_title": path.stem,
            "timestamp": now,
            "document_hash": doc_hash,
            "doc_text_hash": sha256_text(text),
            "namespace": self.namespace,
            "version": int(path.stat().st_mtime_ns),
            "file_size_bytes": path.stat().st_size,
            "doc_id": sha256_text(f"{source}::{doc_hash}"),
        }

    def _deduplicate(self, docs: list[DocumentRecord]) -> list[DocumentRecord]:
        seen: set[tuple[str, str]] = set()
        deduped: list[DocumentRecord] = []
        for doc in docs:
            key = (
                str(doc.metadata.get("document_hash", "")),
                str(doc.metadata.get("source", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(doc)
        return deduped

    def _load_pdf(self, path: Path) -> list[DocumentRecord]:
        try:
            reader = PdfReader(str(path))
        except (PdfReadError, Exception):
            return []

        source = self._relative_path(path)
        records: list[DocumentRecord] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = (page.extract_text() or "").strip()
            except Exception:  # noqa: BLE001
                text = ""
            if not text:
                continue
            cleaned = self._clean_text(text)
            metadata = self._base_metadata(path, cleaned, source_type="pdf") | {
                "page_number": page_number,
                "doc_id": sha256_text(f"{source}::page::{page_number}::{sha256_text(cleaned)}"),
            }
            records.append(DocumentRecord(doc_id=str(metadata["doc_id"]), text=cleaned, metadata=metadata))
        return records

    def _load_text(self, path: Path) -> list[DocumentRecord]:
        text = self._clean_text(path.read_text(encoding="utf-8", errors="replace"))
        if not text:
            return []
        metadata = self._base_metadata(path, text, source_type="txt") | {"page_number": None}
        return [DocumentRecord(doc_id=str(metadata["doc_id"]), text=text, metadata=metadata)]

    def _load_markdown(self, path: Path) -> list[DocumentRecord]:
        text = path.read_text(encoding="utf-8", errors="replace")
        cleaned = self._clean_markdown(text)
        if not cleaned:
            return []
        metadata = self._base_metadata(path, cleaned, source_type="markdown") | {"page_number": None}
        return [DocumentRecord(doc_id=str(metadata["doc_id"]), text=cleaned, metadata=metadata)]

    def _load_html(self, path: Path) -> list[DocumentRecord]:
        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        text = markdownify(str(soup), heading_style="ATX")
        cleaned = self._clean_markdown(text)
        if not cleaned:
            return []
        metadata = self._base_metadata(path, cleaned, source_type="html") | {"page_number": None}
        return [DocumentRecord(doc_id=str(metadata["doc_id"]), text=cleaned, metadata=metadata)]

    def _load_docx(self, path: Path) -> list[DocumentRecord]:
        document = DocxDocument(str(path))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        cleaned = self._clean_text(text)
        if not cleaned:
            return []
        metadata = self._base_metadata(path, cleaned, source_type="docx") | {"page_number": None}
        return [DocumentRecord(doc_id=str(metadata["doc_id"]), text=cleaned, metadata=metadata)]

    def _load_image(self, path: Path) -> list[DocumentRecord]:
        text = self._clean_text(self.ocr_backend.extract_text(path))
        if not text:
            return []
        metadata = self._base_metadata(path, text, source_type="image_ocr") | {"page_number": None}
        return [DocumentRecord(doc_id=str(metadata["doc_id"]), text=text, metadata=metadata)]

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
