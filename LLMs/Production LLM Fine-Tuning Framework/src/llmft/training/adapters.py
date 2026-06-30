"""Adapter management utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llmft.utils.io import ensure_dir, write_json


@dataclass(slots=True)
class AdapterRecord:
    """Adapter metadata record."""

    name: str
    path: Path
    size_bytes: int


class AdapterManager:
    """Manage adapter artifacts for compare/stack/merge operations."""

    def __init__(self, root: str | Path) -> None:
        self.root = ensure_dir(root)

    def register_adapter(self, name: str, content: str) -> AdapterRecord:
        """Create placeholder adapter artifact and metadata."""
        path = self.root / f"{name}.safetensors"
        path.write_text(content, encoding="utf-8")
        return AdapterRecord(name=name, path=path, size_bytes=path.stat().st_size)

    def compare(self, adapters: list[AdapterRecord]) -> Path:
        """Write adapter size comparison report."""
        report_path = self.root / "adapter_comparison.json"
        payload = {
            "adapters": [
                {
                    "name": adapter.name,
                    "path": str(adapter.path),
                    "size_bytes": adapter.size_bytes,
                }
                for adapter in adapters
            ]
        }
        write_json(report_path, payload)
        return report_path

    def stack_manifest(self, adapters: list[AdapterRecord]) -> Path:
        """Write adapter stacking manifest."""
        out = self.root / "adapter_stack_manifest.json"
        write_json(
            out,
            {
                "stack_order": [adapter.name for adapter in adapters],
                "paths": [str(adapter.path) for adapter in adapters],
            },
        )
        return out
