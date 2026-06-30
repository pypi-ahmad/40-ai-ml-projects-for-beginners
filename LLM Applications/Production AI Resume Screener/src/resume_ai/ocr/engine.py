"""OCR extraction for image and scanned PDF content."""

from __future__ import annotations

from pathlib import Path

import cv2
import fitz
import numpy as np
import pytesseract

from resume_ai.config.loader import AppConfig


class OCREngine:
    """OCR engine with Tesseract primary and optional Paddle adapter."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._paddle = None

    def _ensure_paddle(self) -> None:
        if not self.config.ocr.enable_paddle or self._paddle is not None:
            return
        try:
            from paddleocr import PaddleOCR  # type: ignore

            self._paddle = PaddleOCR(use_angle_cls=True, lang="en")
        except Exception:
            self._paddle = None

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        _, threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold

    def extract_from_image(self, path: Path) -> str:
        image = cv2.imread(str(path))
        if image is None:
            return ""

        processed = self.preprocess_image(image)
        text = pytesseract.image_to_string(processed, lang=self.config.ocr.tesseract_lang)
        if text.strip():
            return text

        self._ensure_paddle()
        if self._paddle is None:
            return ""

        output = self._paddle.ocr(str(path), cls=True)
        lines: list[str] = []
        for row in output:
            for item in row:
                lines.append(item[1][0])
        return "\n".join(lines)

    def extract_from_pdf(self, path: Path) -> str:
        doc = fitz.open(path)
        chunks: list[str] = []
        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(dpi=self.config.ocr.dpi)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            text = pytesseract.image_to_string(image, lang=self.config.ocr.tesseract_lang)
            chunks.append(text)
        doc.close()
        return "\n".join(chunks)
