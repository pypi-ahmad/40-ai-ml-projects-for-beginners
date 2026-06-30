"""Model export manager."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from llmft.config.schemas import ExportConfig
from llmft.training.types import TrainingReport
from llmft.utils.io import ensure_dir, write_json


class ExportManager:
    """Create export artifacts and manifests."""

    def __init__(self, config: ExportConfig) -> None:
        self.config = config
        self.root = ensure_dir(config.export_dir)

    def run(self, report: TrainingReport) -> Path:
        """Execute export pipeline and return summary path."""
        run_dir = ensure_dir(self.root / report.run_id)
        outputs: dict[str, str] = {}

        adapter_path = run_dir / "adapter.safetensors"
        adapter_path.write_text("placeholder adapter weights", encoding="utf-8")
        outputs["adapter"] = str(adapter_path)

        if self.config.emit_merged_manifest:
            merged_path = run_dir / "merged_model_manifest.json"
            write_json(
                merged_path,
                {
                    "base_model": report.model_id,
                    "adapter": str(adapter_path),
                    "merged_output": f"{run_dir}/merged",
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
            outputs["merged_manifest"] = str(merged_path)

        if self.config.emit_gguf:
            gguf_path = run_dir / "model.gguf"
            gguf_path.write_text("placeholder gguf binary manifest", encoding="utf-8")
            outputs["gguf"] = str(gguf_path)

        if self.config.emit_ollama_modelfile:
            modelfile_path = run_dir / "Modelfile"
            source_ref = outputs.get("gguf", "./model.gguf")
            modelfile_path.write_text(
                f"FROM {source_ref}\nPARAMETER temperature 0.2\nTEMPLATE \"{{{{ .Prompt }}}}\"\n",
                encoding="utf-8",
            )
            outputs["ollama_modelfile"] = str(modelfile_path)

        if self.config.emit_onnx:
            onnx_path = run_dir / "model.onnx"
            onnx_path.write_text("placeholder onnx", encoding="utf-8")
            outputs["onnx"] = str(onnx_path)

        summary_path = run_dir / "export_summary.json"
        write_json(summary_path, {"run_id": report.run_id, "outputs": outputs})
        return summary_path
