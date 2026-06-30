"""Document readers and scanned-vs-digital detection."""

from __future__ import annotations

import hashlib
from pathlib import Path

import fitz
from docx import Document

from resume_ai.config.loader import AppConfig
from resume_ai.models import OCRMode
from resume_ai.ocr.engine import OCREngine

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class ResumeReader:
    """Read candidate files and route to OCR when needed."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.ocr = OCREngine(config)

    @staticmethod
    def compute_file_hash(path: Path) -> str:
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    @staticmethod
    def _pdf_text_density(path: Path, pages: int = 2) -> float:
        doc = fitz.open(path)
        sample_pages = min(pages, doc.page_count)
        text_chars = 0
        area = 0.0
        for index in range(sample_pages):
            page = doc.load_page(index)
            text_chars += len(page.get_text("text"))
            rect = page.rect
            area += rect.width * rect.height
        doc.close()
        if area == 0:
            return 0.0
        return text_chars / area

    def read(self, path: Path) -> tuple[str, OCRMode]:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")

        if suffix == ".txt" or suffix == ".md":
            return path.read_text(encoding="utf-8", errors="ignore"), OCRMode.DIGITAL

        if suffix == ".docx":
            doc = Document(str(path))
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
            return text, OCRMode.DIGITAL

        if suffix in IMAGE_EXTENSIONS:
            return self.ocr.extract_from_image(path), OCRMode.SCANNED

        density = self._pdf_text_density(path)
        if density > 0.002:
            doc = fitz.open(path)
            text = "\n".join(doc.load_page(i).get_text("text") for i in range(doc.page_count))
            doc.close()
            return text, OCRMode.DIGITAL

        return self.ocr.extract_from_pdf(path), OCRMode.SCANNED
