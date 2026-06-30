"""Async inference engine with prediction caching."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass(slots=True)
class Prediction:
    label_id: int
    label_name: str
    confidence: float


class InferenceEngine:
    """Loads models and provides async inference methods."""

    def __init__(
        self,
        model_path: str,
        label_names: list[str],
        cache_size: int = 2048,
        device: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self.model_path = model_path
        self.label_names = label_names
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model.to(self.device)
        self.model.eval()
        self.max_retries = max_retries

        self._predict_cached = lru_cache(maxsize=cache_size)(self._predict_sync)
        logger.info(f"Inference engine loaded model={model_path} device={self.device}")

    def _predict_sync(self, text: str, top_k: int) -> list[Prediction]:
        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                with torch.no_grad():
                    logits = self.model(**encoded).logits
                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                break
            except Exception as exc:  # pragma: no cover - runtime safeguard
                last_error = exc
                continue
        else:
            raise RuntimeError(f"Inference failed after retries: {last_error}")

        top_idxs = np.argsort(probs)[::-1][:top_k]
        preds = [
            Prediction(
                label_id=int(idx),
                label_name=self.label_names[idx] if idx < len(self.label_names) else str(idx),
                confidence=float(probs[idx]),
            )
            for idx in top_idxs
        ]
        return preds

    async def predict(self, text: str, top_k: int = 3) -> tuple[list[Prediction], float]:
        """Run async single-text prediction and return latency."""
        start = time.perf_counter()
        predictions = await asyncio.to_thread(self._predict_cached, text, top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        return predictions, latency_ms

    async def predict_batch(self, texts: list[str], top_k: int = 3) -> tuple[list[list[Prediction]], float]:
        """Run async batch prediction."""
        start = time.perf_counter()
        tasks = [asyncio.to_thread(self._predict_cached, text, top_k) for text in texts]
        predictions = await asyncio.gather(*tasks)
        latency_ms = (time.perf_counter() - start) * 1000
        return predictions, latency_ms
