"""Typed config contracts for the framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime settings shared across modules."""

    seed: int = 42
    device: str = "auto"
    mixed_precision: str = "bf16"
    artifacts_dir: str = "artifacts"
    cache_dir: str = ".cache/llmft"


@dataclass(slots=True)
class DataConfig:
    """Dataset pipeline settings."""

    datasets: list[str] = field(
        default_factory=lambda: ["alpaca_cleaned", "codealpaca", "medical_qa"]
    )
    streaming: bool = False
    validation_ratio: float = 0.1
    max_samples_per_dataset: int = 500
    min_tokens: int = 4
    max_tokens: int = 2048
    template: str = "alpaca"


@dataclass(slots=True)
class ModelConfig:
    """Model and fallback resolution settings."""

    targets: list[str] = field(
        default_factory=lambda: ["llama3_8b", "qwen3_8b", "mistral_7b", "gemma3", "phi4_mini", "granite41"]
    )
    allow_fallback: bool = True
    dtype: str = "bf16"
    load_in_4bit: bool = True
    quantization_modes: list[str] = field(default_factory=lambda: ["4bit", "8bit", "fp16", "bf16"])


@dataclass(slots=True)
class TrainConfig:
    """Training and optimization settings."""

    method: str = "sft"
    peft_method: str = "qlora"
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation: int = 8
    epochs: int = 1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_seq_len: int = 2048
    eval_steps: int = 50
    save_steps: int = 50
    early_stopping_patience: int = 3


@dataclass(slots=True)
class HPOConfig:
    """Hyperparameter optimization settings."""

    enabled: bool = True
    trials: int = 8
    metric: str = "eval_loss"
    direction: str = "minimize"


@dataclass(slots=True)
class EvalConfig:
    """Evaluation settings."""

    metrics: list[str] = field(
        default_factory=lambda: [
            "bleu",
            "rouge",
            "bertscore",
            "exact_match",
            "perplexity",
            "latency_ms",
        ]
    )
    judge_model: str = "granite41"
    judge_enabled: bool = True


@dataclass(slots=True)
class InferenceConfig:
    """Inference backend settings."""

    backend: str = "transformers"
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40
    ollama_host: str = "http://localhost:11434"
    vllm_host: str = "http://localhost:8000"
    request_timeout_seconds: int = 2
    enable_remote_backends: bool = False
    benchmark_backends: list[str] = field(default_factory=lambda: ["transformers"])


@dataclass(slots=True)
class ExportConfig:
    """Export targets and options."""

    export_dir: str = "artifacts/exports"
    emit_gguf: bool = True
    emit_onnx: bool = False
    emit_ollama_modelfile: bool = True
    emit_merged_manifest: bool = True


@dataclass(slots=True)
class ServeConfig:
    """Serving settings."""

    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1


@dataclass(slots=True)
class UIConfig:
    """Streamlit UI settings."""

    host: str = "0.0.0.0"
    port: int = 8501
    title: str = "LLM Fine-Tuning Framework"


@dataclass(slots=True)
class SafetyConfig:
    """Input/output safety settings."""

    enable_prompt_injection_check: bool = True
    enable_toxicity_check: bool = True
    banned_patterns: list[str] = field(
        default_factory=lambda: ["ignore previous instructions", "system prompt", "api key", "password"]
    )


@dataclass(slots=True)
class ProjectConfig:
    """Top-level project config contract."""

    name: str = "project22-llmft"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    def artifacts_path(self) -> Path:
        """Return resolved artifacts path."""
        return Path(self.runtime.artifacts_dir)
