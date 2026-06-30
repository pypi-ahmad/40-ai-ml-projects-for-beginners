"""ONNX export and quantization utilities."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from loguru import logger
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class ONNXExporter:
    """Export and benchmark transformer sequence classifiers in ONNX."""

    @staticmethod
    def export(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        output_path: str | Path,
        opset: int = 17,
        max_length: int = 256,
    ) -> Path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model.eval()
        effective_opset = max(opset, 18)

        sample = tokenizer(
            "ONNX export sample input.",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        with torch.no_grad():
            torch.onnx.export(
                model,
                args=(sample["input_ids"], sample["attention_mask"]),
                f=str(out_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch"},
                },
                opset_version=effective_opset,
                dynamo=False,
            )
        return out_path

    @staticmethod
    def quantize_dynamic(onnx_path: str | Path, quantized_path: str | Path) -> Path:
        src = Path(onnx_path)
        dst = Path(quantized_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            quantize_dynamic(src.as_posix(), dst.as_posix(), weight_type=QuantType.QInt8)
        except Exception as exc:  # pragma: no cover - environment/model specific
            logger.warning(f"ONNX dynamic quantization failed; using non-quantized export. reason={exc}")
            shutil.copy2(src, dst)
        return dst


def create_session(onnx_path: str | Path) -> ort.InferenceSession:
    """Create ONNX Runtime session with GPU fallback to CPU."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        return ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception:
        return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def benchmark_onnx_latency(session: ort.InferenceSession, inputs: dict[str, np.ndarray], repeats: int = 100) -> float:
    """Measure mean ONNX inference latency in milliseconds."""
    timings_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = session.run(None, inputs)
        timings_ms.append((time.perf_counter() - start) * 1000)
    return float(np.mean(timings_ms))
