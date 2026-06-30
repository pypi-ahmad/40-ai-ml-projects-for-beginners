"""Pydantic schemas for end-to-end experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PathsConfig(BaseModel):
    project_root: Path = Field(default=Path("."))
    data_dir: Path = Field(default=Path("data"))
    artifacts_dir: Path = Field(default=Path("artifacts"))
    logs_dir: Path = Field(default=Path("logs"))
    mlruns_dir: Path = Field(default=Path("mlruns"))


class DatasetConfig(BaseModel):
    source: Literal["hf", "csv", "json", "parquet"] = "hf"
    name: str = "ag_news"
    subset: str | None = None
    text_column: str = "text"
    label_column: str = "label"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    local_path: Path | None = None
    streaming: bool = False
    sample_cap: int | None = 50_000
    min_text_length: int = 1
    deduplicate: bool = True
    stratify_split: bool = True
    split_seed: int = 42
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1)


class TokenizerConfig(BaseModel):
    name: str = "distilbert-base-uncased"
    max_length: int = 256
    truncation: bool = True
    padding: Literal["max_length", "longest", "do_not_pad"] = "max_length"
    dynamic_padding: bool = True


class ModelConfig(BaseModel):
    name: str = "distilbert-base-uncased"
    task: Literal[
        "binary",
        "multiclass",
        "multilabel",
        "hierarchical",
        "zero_shot",
        "few_shot",
        "instruction_tuning",
    ] = "multiclass"
    num_labels: int = 4
    id2label: dict[int, str] | None = None
    label2id: dict[str, int] | None = None


class TrainingConfig(BaseModel):
    engine: Literal["trainer", "accelerate"] = "trainer"
    output_dir: Path = Field(default=Path("artifacts/checkpoints"))
    epochs: int = 2
    learning_rate: float = 2e-5
    train_batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    scheduler: str = "linear"
    logging_steps: int = 25
    eval_steps: int = 200
    save_steps: int = 200
    early_stopping_patience: int = 2
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True
    seed: int = 42


class PeftConfigModel(BaseModel):
    enabled: bool = False
    strategy: Literal["lora", "qlora", "none"] = "none"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: Literal["none", "all", "lora_only"] = "none"


class HpoConfig(BaseModel):
    enabled: bool = False
    trials: int = 20
    timeout_minutes: int = 120
    metric_name: str = "eval_f1"
    direction: Literal["maximize", "minimize"] = "maximize"


class EvalConfig(BaseModel):
    average: Literal["macro", "micro", "weighted", "binary"] = "macro"
    threshold: float = 0.5
    calibration_bins: int = 10


class ExportConfig(BaseModel):
    onnx: bool = True
    torchscript: bool = True
    dynamic_quantization: bool = True
    safetensors: bool = True
    int8: bool = True
    int4: bool = False


class ServeConfig(BaseModel):
    host: str = "0.0.0.0"
    api_port: int = 8000
    ui_port: int = 8501
    workers: int = 1


class MonitoringConfig(BaseModel):
    collect_gpu: bool = True
    interval_seconds: float = 1.0


class ExperimentConfig(BaseModel):
    experiment_name: str = "domain_llm_ft"
    run_name: str = "baseline"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    peft: PeftConfigModel = Field(default_factory=PeftConfigModel)
    hpo: HpoConfig = Field(default_factory=HpoConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    serving: ServeConfig = Field(default_factory=ServeConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @model_validator(mode="after")
    def validate_splits(self) -> "ExperimentConfig":
        split_sum = sum(self.dataset.split_ratio)
        if abs(split_sum - 1.0) > 1e-6:
            msg = "dataset.split_ratio must sum to 1.0"
            raise ValueError(msg)
        return self
