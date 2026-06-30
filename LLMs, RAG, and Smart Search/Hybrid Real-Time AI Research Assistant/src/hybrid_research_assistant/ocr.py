"""OCR backend adapters for document ingestion."""

from __future__ import annotations

import base64
from pathlib import Path

from ollama import Client


class OllamaOCRBackend:
    """OCR backend using a local Ollama multimodal model."""

    def __init__(self, host: str, model: str) -> None:
        self.client = Client(host=host)
        self.model = model

    def extract_text(self, file_path: Path) -> str:
        """Extract text by sending image payload to OCR model."""

        image_bytes = file_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        response = self.client.generate(
            model=self.model,
            prompt="Extract all readable text exactly. Return plain text only.",
            images=[image_b64],
            stream=False,
            options={"temperature": 0.0},
            raw=False,
        )
        return str(response.get("response", "")).strip()

    def close(self) -> None:
        """Close underlying HTTP client resources."""

        self.client.close()


class PytesseractOCRBackend:
    """OCR backend using local pytesseract binary."""

    def __init__(self) -> None:
        try:
            import pytesseract  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
        except Exception as err:  # noqa: BLE001
            raise RuntimeError("pytesseract/Pillow not installed") from err
        self._pytesseract = pytesseract
        self._image = Image

    def extract_text(self, file_path: Path) -> str:
        image = self._image.open(file_path)
        return str(self._pytesseract.image_to_string(image)).strip()
