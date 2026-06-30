"""Document extraction and Q&A pipeline with PDF + OCR support."""

from __future__ import annotations

import base64
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from pdf2image import convert_from_path
from PIL import Image, UnidentifiedImageError
from PyPDF2 import PdfReader

from src.config import get_config
from src.ollama_client import OllamaClient
from src.schemas import DocumentResult, ErrorInfo

logger = logging.getLogger(__name__)


ALLOWED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


class DocumentAnalyzer:
    """Extract text from PDFs/images and answer questions over extracted context."""

    def __init__(
        self,
        ocr_model: str | None = None,
        qa_model: str | None = None,
        max_pages: int | None = None,
        client: OllamaClient | None = None,
    ) -> None:
        cfg = get_config()
        self.ocr_model = ocr_model or cfg.ocr_primary_model
        self.ocr_fallback_model = cfg.ocr_fallback_model
        self.qa_model = qa_model or cfg.chat_model
        self.max_pages = max_pages or cfg.document_max_pages
        self.max_file_size_mb = cfg.document_max_file_size_mb
        self.max_image_pixels = cfg.document_max_image_pixels
        self.client = client or OllamaClient()
        self._owns_client = client is None

    def close(self) -> None:
        """Close owned client resources."""

        if self._owns_client:
            self.client.close()

    def _validate_input_file(self, path: Path) -> str | None:
        """Validate extension, existence, and file size before processing."""

        if not path.exists():
            return "Uploaded file path does not exist."

        suffix = path.suffix.lower()
        if suffix not in ALLOWED_DOCUMENT_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_DOCUMENT_EXTENSIONS))
            return f"Unsupported file format '{suffix}'. Allowed: {allowed}."

        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            return (
                f"File too large: {file_size_mb:.2f} MB. "
                f"Maximum supported size is {self.max_file_size_mb} MB."
            )

        return None

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Serialize PIL image to base64 PNG for Ollama multimodal payload."""

        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_file:
            image.save(tmp_file.name, format="PNG")
            return base64.b64encode(Path(tmp_file.name).read_bytes()).decode("utf-8")

    def _ocr_image(self, image: Image.Image, model: str) -> tuple[str, str | None]:
        """Run OCR model on image and return extracted text."""

        b64_image = self._image_to_base64(image)
        prompt = (
            "Extract all visible text from this image exactly as shown. "
            "Preserve line breaks when possible."
        )
        result = self.client.generate(
            model=model,
            prompt=prompt,
            images=[b64_image],
            max_tokens=2200,
            temperature=0.0,
            raw=False,
        )
        return result.get("response", "").strip(), result.get("error")

    @staticmethod
    def _extract_native_pdf_text(reader: PdfReader, page_index: int) -> str:
        """Extract digital text for one PDF page when available."""

        if page_index >= len(reader.pages):
            return ""
        return (reader.pages[page_index].extract_text() or "").strip()

    def _ocr_with_fallback(self, image: Image.Image) -> tuple[str, str, str | None]:
        """OCR with primary model then fallback model when needed."""

        ocr_text, error = self._ocr_image(image=image, model=self.ocr_model)
        if not error:
            return ocr_text, self.ocr_model, None

        if self.ocr_fallback_model == self.ocr_model:
            return "", self.ocr_model, error

        fallback_text, fallback_error = self._ocr_image(image=image, model=self.ocr_fallback_model)
        if fallback_error:
            return "", self.ocr_model, fallback_error
        return fallback_text, self.ocr_fallback_model, None

    def extract_text(self, file_path: str) -> tuple[str, int, str, list[str], str | None]:
        """Extract text from uploaded PDF or image file."""

        path = Path(file_path)
        validation_error = self._validate_input_file(path)
        if validation_error:
            return "", 0, self.ocr_model, [], validation_error

        warnings: list[str] = []
        ocr_model_used = self.ocr_model

        if path.suffix.lower() == ".pdf":
            try:
                reader = PdfReader(str(path))
                page_count = min(len(reader.pages), self.max_pages)
            except Exception as exc:
                return "", 0, self.ocr_model, [], f"PDF read failed: {exc}"

            if page_count == 0:
                return "", 0, self.ocr_model, [], "PDF has no readable pages."

            try:
                rendered_pages = convert_from_path(
                    str(path),
                    first_page=1,
                    last_page=page_count,
                )
            except Exception as exc:
                return "", 0, self.ocr_model, [], f"PDF render failed: {exc}"

            text_chunks: list[str] = []
            for page_index in range(page_count):
                native_text = self._extract_native_pdf_text(reader, page_index)
                if len(native_text) >= 40:
                    text_chunks.append(f"[Page {page_index + 1}]\n{native_text}")
                    continue

                if page_index >= len(rendered_pages):
                    warnings.append(f"Page {page_index + 1}: no rendered image available for OCR.")
                    continue

                ocr_text, used_model, ocr_error = self._ocr_with_fallback(
                    rendered_pages[page_index]
                )
                ocr_model_used = used_model
                if ocr_error:
                    warnings.append(f"Page {page_index + 1}: OCR failed ({ocr_error}).")
                    continue

                if not ocr_text.strip():
                    warnings.append(f"Page {page_index + 1}: OCR produced empty text.")
                    continue

                text_chunks.append(f"[Page {page_index + 1}]\n{ocr_text}")

            extracted = "\n\n".join(text_chunks).strip()
            if not extracted:
                fallback_error = "No text extracted from PDF."
                if warnings:
                    fallback_error += f" OCR warnings: {' | '.join(warnings[:3])}"
                return "", page_count, ocr_model_used, warnings, fallback_error

            return extracted, page_count, ocr_model_used, warnings, None

        try:
            with Image.open(path) as raw_image:
                if raw_image.width * raw_image.height > self.max_image_pixels:
                    return (
                        "",
                        0,
                        self.ocr_model,
                        [],
                        (
                            f"Image resolution too large ({raw_image.width}x{raw_image.height}). "
                            f"Limit is {self.max_image_pixels} pixels."
                        ),
                    )
                image = raw_image.convert("RGB")
        except UnidentifiedImageError:
            return "", 0, self.ocr_model, [], "Uploaded image format is invalid or corrupted."
        except Exception as exc:
            return "", 0, self.ocr_model, [], f"Could not open image: {exc}"

        ocr_text, used_model, ocr_error = self._ocr_with_fallback(image)
        if ocr_error:
            return "", 1, used_model, [], f"OCR failed: {ocr_error}"
        if not ocr_text.strip():
            return "", 1, used_model, [], "OCR completed but no text was detected."

        return ocr_text, 1, used_model, [], None

    def answer_question(self, extracted_text: str, question: str) -> tuple[str, str | None]:
        """Run grounded Q&A over extracted document text."""

        if not question.strip():
            return "", None

        prompt = (
            "Use only document text below to answer question. "
            "If answer not present, say 'Not found in document'.\n\n"
            f"Document:\n{extracted_text[:12000]}\n\n"
            f"Question:\n{question.strip()}"
        )
        result = self.client.generate(
            model=self.qa_model,
            prompt=prompt,
            temperature=0.2,
            max_tokens=700,
        )
        if result["error"]:
            return "", result["error"]
        return result["response"].strip(), None

    def summarize(self, extracted_text: str) -> tuple[str, str | None]:
        """Create concise summary for extracted document text."""

        prompt = (
            "Summarize document in 3 concise bullets and one short paragraph. "
            "Keep factual and grounded in text.\n\n"
            f"Document:\n{extracted_text[:12000]}"
        )
        result = self.client.generate(
            model=self.qa_model,
            prompt=prompt,
            temperature=0.2,
            max_tokens=700,
        )
        if result["error"]:
            return "", result["error"]
        return result["response"].strip(), None

    def analyze_document(self, file_path: str, question: str = "") -> DocumentResult:
        """Run complete document analysis pipeline."""

        started = time.perf_counter()
        (
            extracted_text,
            pages_processed,
            ocr_model_used,
            warnings,
            extraction_error,
        ) = self.extract_text(file_path)

        if extraction_error:
            return DocumentResult(
                extracted_text="",
                summary="",
                answer="",
                pages_processed=pages_processed,
                ocr_model_used=ocr_model_used,
                qa_model=self.qa_model,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                warnings=warnings,
                error=ErrorInfo(message=extraction_error, stage="extraction"),
            )

        summary, summary_error = self.summarize(extracted_text)
        if summary_error:
            warnings.append(f"Summary generation failed: {summary_error}")
            summary = "Summary unavailable due to model error."

        answer = ""
        if question.strip():
            answer, qa_error = self.answer_question(extracted_text, question)
            if qa_error:
                warnings.append(f"Q&A generation failed: {qa_error}")
                answer = "Answer unavailable due to model error."

        return DocumentResult(
            extracted_text=extracted_text[:15000],
            summary=summary,
            answer=answer,
            pages_processed=pages_processed,
            ocr_model_used=ocr_model_used,
            qa_model=self.qa_model,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            warnings=warnings,
            error=None,
        )


def analyze_document(
    file_path: str,
    question: str = "",
    ocr_model: str | None = None,
    qa_model: str | None = None,
    client: OllamaClient | None = None,
) -> dict[str, Any]:
    """Backwards-compatible wrapper returning dictionary payload."""

    analyzer = DocumentAnalyzer(ocr_model=ocr_model, qa_model=qa_model, client=client)
    try:
        return analyzer.analyze_document(file_path=file_path, question=question).model_dump()
    finally:
        analyzer.close()
