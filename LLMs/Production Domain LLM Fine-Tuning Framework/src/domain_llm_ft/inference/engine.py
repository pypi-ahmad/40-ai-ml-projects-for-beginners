"""Inference engines for PyTorch and ONNX backends."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from domain_llm_ft.models.registry import resolve_model_name


@dataclass
class Prediction:
    label: str
    score: float
    probabilities: list[float]


class InferenceEngine:
    """Unified sync/async inference for CPU/GPU."""

    def __init__(self, model_name: str, device: str = "auto"):
        resolved = resolve_model_name(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.model = AutoModelForSequenceClassification.from_pretrained(resolved)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str) -> Prediction:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        label_id = int(np.argmax(probs))
        label = self.model.config.id2label.get(label_id, str(label_id))
        return Prediction(label=label, score=float(probs[label_id]), probabilities=probs.tolist())

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        return [self.predict(text) for text in texts]

    async def predict_async(self, text: str) -> Prediction:
        return await asyncio.to_thread(self.predict, text)


class OnnxInferenceEngine:
    """ONNX Runtime inference backend."""

    def __init__(self, model_path: Path, tokenizer_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(resolve_model_name(tokenizer_name))
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)

    def predict(self, text: str) -> Prediction:
        encoded = self.tokenizer(text, return_tensors="np", truncation=True)
        outputs = self.session.run(None, {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        })
        logits = outputs[0]
        probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        probs = probs[0]
        idx = int(np.argmax(probs))
        return Prediction(label=str(idx), score=float(probs[idx]), probabilities=probs.tolist())
