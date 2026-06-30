import base64
from pathlib import Path
from typing import cast

from src.ollama_client import OllamaClient


class DocumentAnalyzer:
    def __init__(self, ocr_model: str = "glm-ocr", qa_model: str = "qwen3.5:4b") -> None:
        self.ocr_model = ocr_model
        self.qa_model = qa_model
        self._client = OllamaClient()

    def extract_text(self, image_path: str | Path) -> str:
        path = Path(image_path)
        if not path.exists():
            return f"File not found: {image_path}"
        b64 = base64.b64encode(path.read_bytes()).decode()
        prompt = "Extract all text from this image."
        result = self._client.generate(
            self.ocr_model,
            prompt,
            raw=False,
            temperature=0.0,
            images=[b64],
        )
        raw = result.get("response", "")
        return raw.strip() or "No text extracted."

    def answer_question(self, context: str, question: str) -> str:
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely:"
        result = self._client.generate(self.qa_model, prompt, temperature=0.2)
        return cast(str, result.get("response", "").strip())

    def close(self) -> None:
        self._client.close()
