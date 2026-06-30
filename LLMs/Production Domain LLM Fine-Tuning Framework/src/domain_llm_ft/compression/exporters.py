"""Model compression and export utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file


class _SequenceClassifierWrapper(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export_torchscript(model, tokenizer, output_dir: Path) -> Path:
    """Export model to TorchScript."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dummy = tokenizer("Hello world", return_tensors="pt")
    model.eval()
    wrapper = _SequenceClassifierWrapper(model)
    traced = torch.jit.trace(wrapper, (dummy["input_ids"], dummy.get("attention_mask")))
    path = output_dir / "model.ts"
    traced.save(str(path))
    return path


def export_onnx(model, tokenizer, output_dir: Path) -> Path:
    """Export model to ONNX."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = tokenizer("Hello world", return_tensors="pt")
    wrapper = _SequenceClassifierWrapper(model)
    path = output_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        (dummy["input_ids"], dummy.get("attention_mask")),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    return path


def export_safetensors(model, output_dir: Path) -> Path:
    """Export model weights in safetensors format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "model.safetensors"
    save_file(model.state_dict(), str(path))
    return path


def dynamic_quantize(model):
    """Apply dynamic quantization for CPU inference."""
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
