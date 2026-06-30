"""Model and adapter export helpers."""

from __future__ import annotations

from pathlib import Path
import shutil

from peft_platform.utils.io import ensure_dir


class Exporter:
    """Export adapters and merged models to target formats."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        ensure_dir(output_root)

    def export_adapter(self, source_dir: Path, name: str) -> Path:
        target = self.output_root / f"{name}.safetensors"
        if source_dir.is_file():
            shutil.copy2(source_dir, target)
        else:
            ensure_dir(target.parent)
            target.write_text("adapter export placeholder", encoding="utf-8")
        return target

    def export_gguf(self, source_dir: Path, name: str) -> Path:
        target = self.output_root / f"{name}.gguf"
        target.write_text(f"GGUF export placeholder from {source_dir}", encoding="utf-8")
        return target

    def export_onnx(self, source_dir: Path, name: str) -> Path:
        target = self.output_root / f"{name}.onnx"
        target.write_text(f"ONNX export placeholder from {source_dir}", encoding="utf-8")
        return target
