"""Adapter management service."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from peft_platform.utils.io import ensure_dir, read_json, write_json


@dataclass(slots=True)
class AdapterRecord:
    name: str
    method: str
    base_model: str
    path: str
    merged: bool = False


class AdapterManager:
    """Manage adapter registry and local artifacts."""

    def __init__(self, registry_path: Path) -> None:
        self.registry_path = registry_path
        ensure_dir(registry_path.parent)
        if not registry_path.exists():
            write_json(registry_path, {"adapters": []})

    def list_adapters(self) -> list[AdapterRecord]:
        payload = read_json(self.registry_path)
        return [AdapterRecord(**record) for record in payload.get("adapters", [])]

    def add_adapter(self, record: AdapterRecord) -> None:
        payload = read_json(self.registry_path)
        adapters = payload.get("adapters", [])
        adapters = [item for item in adapters if item["name"] != record.name]
        adapters.append(asdict(record))
        write_json(self.registry_path, {"adapters": adapters})

    def remove_adapter(self, name: str) -> bool:
        payload = read_json(self.registry_path)
        adapters = payload.get("adapters", [])
        filtered = [item for item in adapters if item["name"] != name]
        changed = len(filtered) != len(adapters)
        if changed:
            write_json(self.registry_path, {"adapters": filtered})
        return changed

    def merge_adapter(self, name: str, merged_path: Path) -> AdapterRecord:
        adapters = self.list_adapters()
        for adapter in adapters:
            if adapter.name == name:
                merged = AdapterRecord(
                    name=f"{name}_merged",
                    method=adapter.method,
                    base_model=adapter.base_model,
                    path=str(merged_path),
                    merged=True,
                )
                self.add_adapter(merged)
                return merged
        raise KeyError(f"Adapter not found: {name}")
