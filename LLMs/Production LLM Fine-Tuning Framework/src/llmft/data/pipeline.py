"""Configurable dataset pipeline for instruction tuning."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llmft.config.schemas import DataConfig
from llmft.templates.registry import TemplateRegistry

_DATASET_SOURCES: dict[str, tuple[str, str | None]] = {
    "alpaca_cleaned": ("yahma/alpaca-cleaned", None),
    "codealpaca": ("sahil2801/CodeAlpaca-20k", None),
    "medical_qa": ("lavita/medical-qa-datasets", "medical_meadow_medical_flashcards"),
}


@dataclass(slots=True)
class DatasetSample:
    """Normalized dataset sample."""

    instruction: str
    input_text: str
    output_text: str
    formatted_text: str


@dataclass(slots=True)
class DatasetStats:
    """Dataset profile statistics."""

    rows_total: int
    rows_after_filter: int
    rows_train: int
    rows_validation: int
    unique_ratio: float
    avg_tokens: float
    p95_tokens: int


@dataclass(slots=True)
class DatasetBundle:
    """Dataset outputs and metadata."""

    train: list[DatasetSample]
    validation: list[DatasetSample]
    stats: DatasetStats
    version_id: str
    manifest_path: Path


class DatasetPipeline:
    """Build cached dataset bundles for training/evaluation."""

    def __init__(self, cache_dir: str | Path, templates: TemplateRegistry | None = None, seed: int = 42) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.templates = templates or TemplateRegistry()
        self.seed = seed
        self._configure_hf_cache()

    def build(self, config: DataConfig, artifacts_dir: str | Path) -> DatasetBundle:
        """Build normalized dataset bundle.

        Args:
            config: Dataset configuration.
            artifacts_dir: Root artifacts directory.

        Returns:
            Dataset bundle with split datasets and stats.
        """
        raw_records = self._load_records(config)
        filtered = self._filter_records(raw_records, config)
        deduped = self._deduplicate(filtered)
        samples = self._format_records(deduped, template_name=config.template)
        train, validation = self._split(samples, config.validation_ratio)
        stats = self._profile(raw_records_count=len(raw_records), samples=samples, train=train, validation=validation)

        version_id = self._version_id(config, stats.rows_after_filter)
        manifest_path = self._persist(artifacts_dir, version_id, config, train, validation, stats)
        return DatasetBundle(train=train, validation=validation, stats=stats, version_id=version_id, manifest_path=manifest_path)

    def _configure_hf_cache(self) -> None:
        hf_home = self.cache_dir / "huggingface"
        datasets_cache = hf_home / "datasets"
        hub_cache = hf_home / "hub"
        transformers_cache = hf_home / "transformers"
        for path in (hf_home, datasets_cache, hub_cache, transformers_cache):
            path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))

    def _load_records(self, config: DataConfig) -> list[dict[str, str]]:
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("datasets package is required for real data loading") from exc

        records: list[dict[str, str]] = []
        for dataset_name in config.datasets:
            dataset_id, dataset_config = self._resolve_dataset_source(dataset_name)
            split_spec = "train" if config.streaming else f"train[:{config.max_samples_per_dataset}]"
            dataset = load_dataset(
                dataset_id,
                dataset_config,
                split=split_spec,
                streaming=config.streaming,
                cache_dir=os.environ.get("HF_DATASETS_CACHE"),
            )

            if config.streaming:
                iterator = iter(dataset)
                for idx, row in enumerate(iterator):
                    if idx >= config.max_samples_per_dataset:
                        break
                    normalized = self._normalize_row(row, dataset_name)
                    if normalized is not None:
                        records.append(normalized)
            else:
                for row in dataset:
                    normalized = self._normalize_row(row, dataset_name)
                    if normalized is not None:
                        records.append(normalized)

        if not records:
            raise RuntimeError("No records loaded from configured datasets")
        return records

    def _resolve_dataset_source(self, dataset_name: str) -> tuple[str, str | None]:
        if dataset_name in _DATASET_SOURCES:
            return _DATASET_SOURCES[dataset_name]
        if dataset_name.startswith("hf:"):
            dataset_id = dataset_name.split("hf:", 1)[1]
            if "::" in dataset_id:
                base, cfg = dataset_id.split("::", 1)
                return base, cfg
            return dataset_id, None
        raise KeyError(f"Unsupported dataset key: {dataset_name}")

    def _normalize_row(self, row: dict[str, Any], dataset_name: str) -> dict[str, str] | None:
        if {"instruction", "input", "output"}.issubset(row.keys()):
            return {
                "instruction": str(row.get("instruction", "")).strip(),
                "input": str(row.get("input", "")).strip(),
                "output": str(row.get("output", "")).strip(),
            }

        if {"prompt", "completion"}.issubset(row.keys()):
            return {
                "instruction": str(row.get("prompt", "")).strip(),
                "input": "",
                "output": str(row.get("completion", "")).strip(),
            }

        if {"question", "answer"}.issubset(row.keys()):
            return {
                "instruction": str(row.get("question", "")).strip(),
                "input": "",
                "output": str(row.get("answer", "")).strip(),
            }

        if "messages" in row and isinstance(row["messages"], list):
            user_text = ""
            assistant_text = ""
            for message in row["messages"]:
                role = str(message.get("role", "")).lower()
                content = str(message.get("content", "")).strip()
                if role == "user" and not user_text:
                    user_text = content
                if role == "assistant" and not assistant_text:
                    assistant_text = content
            if user_text and assistant_text:
                return {
                    "instruction": user_text,
                    "input": "",
                    "output": assistant_text,
                }

        text_fields = [str(value).strip() for value in row.values() if isinstance(value, str) and value.strip()]
        if len(text_fields) >= 2:
            return {
                "instruction": text_fields[0],
                "input": "",
                "output": text_fields[1],
            }

        _ = dataset_name
        return None

    def _filter_records(self, records: list[dict[str, str]], config: DataConfig) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for row in records:
            token_count = len(f"{row['instruction']} {row['input']} {row['output']}".split())
            if token_count < config.min_tokens:
                continue
            if token_count > config.max_tokens:
                continue
            out.append(row)
        return out

    def _deduplicate(self, records: list[dict[str, str]]) -> list[dict[str, str]]:
        seen: set[str] = set()
        deduped: list[dict[str, str]] = []
        for row in records:
            key = hashlib.sha1(
                f"{row['instruction'].strip().lower()}|{row['input'].strip().lower()}|{row['output'].strip().lower()}".encode("utf-8")
            ).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped

    def _format_records(self, records: list[dict[str, str]], template_name: str) -> list[DatasetSample]:
        samples: list[DatasetSample] = []
        for row in records:
            rendered = self.templates.render(template_name, row["instruction"], row["input"], row["output"])
            samples.append(
                DatasetSample(
                    instruction=row["instruction"],
                    input_text=row["input"],
                    output_text=row["output"],
                    formatted_text=rendered,
                )
            )
        return samples

    def _split(self, samples: list[DatasetSample], validation_ratio: float) -> tuple[list[DatasetSample], list[DatasetSample]]:
        rng = random.Random(self.seed)
        cloned = list(samples)
        rng.shuffle(cloned)
        val_size = max(1, int(len(cloned) * validation_ratio)) if cloned else 0
        validation = cloned[:val_size]
        train = cloned[val_size:]
        return train, validation

    def _profile(
        self,
        raw_records_count: int,
        samples: list[DatasetSample],
        train: list[DatasetSample],
        validation: list[DatasetSample],
    ) -> DatasetStats:
        token_lengths = [len(sample.formatted_text.split()) for sample in samples] or [0]
        token_sorted = sorted(token_lengths)
        p95_idx = max(0, int(0.95 * (len(token_sorted) - 1)))
        unique_ratio = len(samples) / raw_records_count if raw_records_count else 0.0
        return DatasetStats(
            rows_total=raw_records_count,
            rows_after_filter=len(samples),
            rows_train=len(train),
            rows_validation=len(validation),
            unique_ratio=round(unique_ratio, 4),
            avg_tokens=round(sum(token_lengths) / len(token_lengths), 2),
            p95_tokens=token_sorted[p95_idx],
        )

    def _version_id(self, config: DataConfig, row_count: int) -> str:
        payload = {
            "datasets": config.datasets,
            "template": config.template,
            "rows": row_count,
            "streaming": config.streaming,
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        return f"dataset-{digest}"

    def _persist(
        self,
        artifacts_dir: str | Path,
        version_id: str,
        config: DataConfig,
        train: list[DatasetSample],
        validation: list[DatasetSample],
        stats: DatasetStats,
    ) -> Path:
        root = Path(artifacts_dir) / "datasets" / version_id
        root.mkdir(parents=True, exist_ok=True)

        train_path = root / "train.jsonl"
        val_path = root / "validation.jsonl"
        manifest_path = root / "manifest.json"

        self._write_jsonl(train_path, train)
        self._write_jsonl(val_path, validation)

        manifest = {
            "version_id": version_id,
            "created_at": datetime.now(UTC).isoformat(),
            "config": asdict(config),
            "stats": asdict(stats),
            "train_path": str(train_path),
            "validation_path": str(val_path),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    def _write_jsonl(self, path: Path, samples: list[DatasetSample]) -> None:
        with path.open("w", encoding="utf-8") as file_obj:
            for sample in samples:
                file_obj.write(json.dumps(asdict(sample), ensure_ascii=True) + "\n")
