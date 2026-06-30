"""Document pipeline tests."""

from __future__ import annotations

from pathlib import Path

from multimodal_ai.adapters.registry import AdapterRegistry
from multimodal_ai.pipelines.document_pipeline import DocumentPipeline


class DummyOCR:
    name = "dummy_ocr"

    def health(self) -> dict:
        return {"ok": True}

    def extract(self, path: str) -> dict:
        return {
            "engine": self.name,
            "text": f"ocr:{Path(path).name}",
            "blocks": [],
            "tables": [],
        }


def test_document_pipeline_image_ocr(tmp_path: Path) -> None:
    image = tmp_path / "sample.png"
    image.write_bytes(b"fake")

    registry = AdapterRegistry()
    registry.register_ocr("dummy", DummyOCR)

    pipeline = DocumentPipeline(registry=registry, primary_engine="dummy", min_text_chars=40)
    result = pipeline.run(str(image))

    assert result.engine == "dummy_ocr"
    assert result.text.startswith("ocr:")
