"""OCR adapter implementations."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image

from multimodal_ai.adapters.base import OCRAdapter


class GLMOcrAdapter(OCRAdapter):
    """OCR adapter using local Ollama multimodal model."""

    def __init__(
        self, model: str = "glm-ocr:latest", base_url: str = "http://localhost:11434"
    ) -> None:
        self.name = "glm_ocr"
        self._model = model
        self._client = httpx.Client(base_url=base_url, timeout=180.0)

    def health(self) -> dict[str, Any]:
        try:
            response = self._client.get("/api/ps")
            response.raise_for_status()
            return {"ok": True, "name": self.name}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "name": self.name, "error": str(exc)}

    def extract(self, path: str) -> dict[str, Any]:
        file_path = Path(path)
        if not file_path.exists():
            return {
                "engine": self.name,
                "text": "",
                "blocks": [],
                "tables": [],
                "error": "missing_file",
            }

        image_b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
        payload = {
            "model": self._model,
            "prompt": "Extract all visible text with layout clues.",
            "images": [image_b64],
            "raw": False,
            "options": {"temperature": 0.0},
        }

        response = self._client.post("/api/generate", json=payload)
        response.raise_for_status()
        text = ""
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            text += chunk.get("response", "")

        return {
            "engine": self.name,
            "text": text.strip(),
            "blocks": [],
            "tables": [],
        }


class EasyOCRAdapter(OCRAdapter):
    """EasyOCR adapter."""

    def __init__(self, lang_list: list[str] | None = None) -> None:
        self.name = "easyocr"
        self._lang_list = lang_list or ["en"]
        self._reader: Any | None = None

    def _load(self) -> None:
        if self._reader is not None:
            return
        try:
            import easyocr

            self._reader = easyocr.Reader(self._lang_list, gpu=False)
        except Exception:  # noqa: BLE001
            self._reader = None

    def health(self) -> dict[str, Any]:
        self._load()
        return {"ok": self._reader is not None, "name": self.name}

    def extract(self, path: str) -> dict[str, Any]:
        self._load()
        if self._reader is None:
            return {"engine": self.name, "text": "", "blocks": [], "tables": []}

        result = self._reader.readtext(path)
        blocks = [
            {
                "text": item[1],
                "bbox": [float(v) for pair in item[0] for v in pair],
                "confidence": float(item[2]),
            }
            for item in result
        ]
        text = "\n".join(block["text"] for block in blocks)
        return {"engine": self.name, "text": text, "blocks": blocks, "tables": []}


class TesseractOCRAdapter(OCRAdapter):
    """Tesseract OCR adapter."""

    def __init__(self) -> None:
        self.name = "tesseract"

    def health(self) -> dict[str, Any]:
        try:
            import pytesseract

            _ = pytesseract.get_tesseract_version()
            return {"ok": True, "name": self.name}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "name": self.name, "error": str(exc)}

    def extract(self, path: str) -> dict[str, Any]:
        try:
            import pytesseract

            image = Image.open(path)
            text = pytesseract.image_to_string(image)
            return {"engine": self.name, "text": text.strip(), "blocks": [], "tables": []}
        except Exception:
            return {"engine": self.name, "text": "", "blocks": [], "tables": []}


class PaddleOCRAdapter(OCRAdapter):
    """PaddleOCR adapter."""

    def __init__(self) -> None:
        self.name = "paddleocr"
        self._ocr: Any | None = None

    def _load(self) -> None:
        if self._ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(use_angle_cls=True, lang="en")
        except Exception:  # noqa: BLE001
            self._ocr = None

    def health(self) -> dict[str, Any]:
        self._load()
        return {"ok": self._ocr is not None, "name": self.name}

    def extract(self, path: str) -> dict[str, Any]:
        self._load()
        if self._ocr is None:
            return {"engine": self.name, "text": "", "blocks": [], "tables": []}

        result = self._ocr.ocr(path, cls=True)
        blocks: list[dict[str, Any]] = []
        for line_group in result:
            for item in line_group:
                bbox = item[0]
                text_info = item[1]
                blocks.append(
                    {
                        "text": text_info[0],
                        "bbox": [float(v) for pair in bbox for v in pair],
                        "confidence": float(text_info[1]),
                    }
                )
        text = "\n".join(block["text"] for block in blocks)
        return {"engine": self.name, "text": text, "blocks": blocks, "tables": []}


def estimate_scanned_document(path: str) -> bool:
    """Heuristic scan detector using grayscale entropy."""

    try:
        image = Image.open(path).convert("L")
        arr = np.array(image)
        unique_ratio = float(len(np.unique(arr))) / 256.0
        return unique_ratio > 0.35
    except Exception:  # noqa: BLE001
        return False
