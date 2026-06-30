"""Typed configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    revision: str = "main"
    trust_remote_code: bool = False
    max_length: int = 2048


@dataclass(slots=True)
class DatasetConfig:
    source: str = "hf"
    name: str = "tatsu-lab/alpaca"
    split: str = "train"
    text_field: str = "text"
    label_field: str = "output"
    val_size: float = 0.1
    test_size: float = 0.1
    seed: int = 42


@dataclass(slots=True)
class PeftConfig:
    method: str = "lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass(slots=True)
class TrainConfig:
    output_dir: str = "artifacts/checkpoints"
    max_steps: int = 200
    epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    grad_accum: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    stop_sequences: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    peft: PeftConfig = field(default_factory=PeftConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    task: str = "instruction_tuning"
    extras: dict[str, Any] = field(default_factory=dict)
