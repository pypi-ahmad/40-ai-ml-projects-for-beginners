from __future__ import annotations

from pathlib import Path

from peft_platform.peft.adapters import AdapterManager, AdapterRecord


def test_adapter_lifecycle(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    manager = AdapterManager(registry)

    record = AdapterRecord(
        name="adapter1",
        method="lora",
        base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        path="artifacts/checkpoints/adapter1",
    )
    manager.add_adapter(record)
    adapters = manager.list_adapters()
    assert len(adapters) == 1
    assert adapters[0].name == "adapter1"

    merged = manager.merge_adapter("adapter1", tmp_path / "merged")
    assert merged.merged

    removed = manager.remove_adapter("adapter1")
    assert removed
