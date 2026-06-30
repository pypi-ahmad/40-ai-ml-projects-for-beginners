"""Unit tests for document analyzer validation behavior."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.document_analyzer import DocumentAnalyzer


class StubClient:
    """Stub client for deterministic OCR and QA responses."""

    def generate(self, **kwargs):
        prompt = kwargs.get("prompt", "")
        if "Extract all visible text" in prompt:
            return {
                "response": "Invoice #42\nTotal: $99",
                "latency_ms": 10.0,
                "eval_count": 30,
                "eval_duration_ns": 100,
                "error": None,
            }
        if "Summarize document" in prompt:
            return {
                "response": "- Invoice detected\n- Total amount present\nSummary paragraph.",
                "latency_ms": 12.0,
                "eval_count": 30,
                "eval_duration_ns": 100,
                "error": None,
            }
        return {
            "response": "Total is 99 dollars.",
            "latency_ms": 8.0,
            "eval_count": 30,
            "eval_duration_ns": 100,
            "error": None,
        }

    def close(self):
        pass


def test_document_analyzer_rejects_missing_file() -> None:
    analyzer = DocumentAnalyzer(client=StubClient())
    result = analyzer.analyze_document("/tmp/file-does-not-exist.pdf")

    assert result.error is not None
    assert result.error.stage == "extraction"


def test_document_analyzer_image_pipeline(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (120, 80), "white").save(image_path)

    analyzer = DocumentAnalyzer(client=StubClient())
    result = analyzer.analyze_document(str(image_path), question="What is total?")

    assert result.error is None
    assert "Invoice" in result.extracted_text
    assert result.pages_processed == 1


def test_document_analyzer_rejects_unsupported_extension(tmp_path: Path) -> None:
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("not a supported document", encoding="utf-8")

    analyzer = DocumentAnalyzer(client=StubClient())
    result = analyzer.analyze_document(str(bad_file))

    assert result.error is not None
    assert "Unsupported file format" in result.error.message


def test_document_analyzer_rejects_oversized_image(tmp_path: Path) -> None:
    image_path = tmp_path / "too_big.png"
    Image.new("RGB", (50, 50), "white").save(image_path)

    analyzer = DocumentAnalyzer(client=StubClient())
    analyzer.max_image_pixels = 1000
    result = analyzer.analyze_document(str(image_path))

    assert result.error is not None
    assert "Image resolution too large" in result.error.message


def test_document_analyzer_rejects_corrupted_image(tmp_path: Path) -> None:
    image_path = tmp_path / "corrupt.png"
    image_path.write_bytes(b"not-a-real-image")

    analyzer = DocumentAnalyzer(client=StubClient())
    result = analyzer.analyze_document(str(image_path))

    assert result.error is not None
    assert "invalid or corrupted" in result.error.message
