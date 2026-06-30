"""Vision-language adapter implementations."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx

from multimodal_ai.adapters.base import VisionModelAdapter


class _FallbackVisionAdapter(VisionModelAdapter):
    """Safe fallback for unavailable vision backends."""

    def __init__(self, name: str) -> None:
        self.name = name

    def health(self) -> dict[str, Any]:
        return {"ok": True, "name": self.name, "mode": "fallback"}

    def caption(self, image_path: str, style: str = "detailed") -> dict[str, Any]:
        path = Path(image_path)
        stem = path.stem or "image"
        templates = {
            "short": f"{stem}: visual scene detected.",
            "detailed": f"{stem}: multimodal fallback caption with limited visual certainty.",
            "social": f"Sharing {stem} with key visual highlights.",
            "technical": (
                f"{stem}: fallback technical description pending "
                "high-fidelity VLM inference."
            ),
            "alt_text": f"Image {stem}. Scene content available via fallback model.",
        }
        return {
            "model": self.name,
            "style": style,
            "caption": templates.get(style, templates["detailed"]),
            "confidence": 0.35,
        }

    def vqa(self, image_path: str, question: str) -> dict[str, Any]:
        return {
            "model": self.name,
            "answer": "Fallback VLM path active. Install local vision model for grounded answer.",
            "confidence": 0.2,
            "evidence": [
                "No active VLM backend detected.",
                f"Question received: {question}",
            ],
        }


class OllamaVisionAdapter(VisionModelAdapter):
    """Vision adapter through Ollama generate endpoint."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        self.name = f"ollama:{model}"
        self._model = model
        self._client = httpx.Client(base_url=base_url, timeout=180.0)

    def health(self) -> dict[str, Any]:
        try:
            response = self._client.get("/api/ps")
            response.raise_for_status()
            return {"ok": True, "name": self.name}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "name": self.name, "error": str(exc)}

    def _call(self, prompt: str, image_path: str) -> str:
        file_path = Path(image_path)
        if not file_path.exists():
            return "Image not found"
        image_b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
        payload = {
            "model": self._model,
            "prompt": prompt,
            "images": [image_b64],
            "raw": False,
            "options": {"temperature": 0.1},
        }
        response = self._client.post("/api/generate", json=payload)
        response.raise_for_status()

        text = ""
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            text += chunk.get("response", "")
        return text.strip()

    def caption(self, image_path: str, style: str = "detailed") -> dict[str, Any]:
        prompt = (
            f"Generate a {style} caption for this image. "
            "Return plain text only, grounded in visible content."
        )
        text = self._call(prompt, image_path)
        return {"model": self.name, "style": style, "caption": text, "confidence": 0.75}

    def vqa(self, image_path: str, question: str) -> dict[str, Any]:
        prompt = (
            f"Question: {question}\n"
            "Answer only from image evidence. Include brief reasoning in one sentence."
        )
        answer = self._call(prompt, image_path)
        return {
            "model": self.name,
            "answer": answer,
            "confidence": 0.72,
            "evidence": ["Vision-language model response"],
        }


class Florence2Adapter(_FallbackVisionAdapter):
    def __init__(self) -> None:
        super().__init__("florence_2")


class Qwen25VLAdapter(_FallbackVisionAdapter):
    def __init__(self) -> None:
        super().__init__("qwen2_5_vl")


class LlamaVisionAdapter(_FallbackVisionAdapter):
    def __init__(self) -> None:
        super().__init__("llama_vision")


class MiniCPMVAdapter(_FallbackVisionAdapter):
    def __init__(self) -> None:
        super().__init__("minicpm_v")


class BLIP2Adapter(_FallbackVisionAdapter):
    def __init__(self) -> None:
        super().__init__("blip2")


class SmolVLMAdapter(_FallbackVisionAdapter):
    def __init__(self) -> None:
        super().__init__("smolvlm")
