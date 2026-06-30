"""Configuration objects for training and benchmarking profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class PathConfig:
    """Filesystem layout used across scripts, notebooks, and app."""

    project_root: Path
    data_dir: Path
    outputs_dir: Path
    checkpoints_dir: Path
    figures_dir: Path
    results_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> "PathConfig":
        project_root = project_root.resolve()
        outputs_dir = project_root / "outputs"
        return cls(
            project_root=project_root,
            data_dir=project_root / "data",
            outputs_dir=outputs_dir,
            checkpoints_dir=outputs_dir / "checkpoints",
            figures_dir=outputs_dir / "figures",
            results_dir=outputs_dir / "results",
        )

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.outputs_dir,
            self.checkpoints_dir,
            self.figures_dir,
            self.results_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class TrainingProfile:
    """Hyperparameters for a run profile.

    Attributes:
        name: Profile name used in artifact filenames.
        max_tokens: Max token budget for training corpus.
        context_len: Context window size for next-word prediction.
        vocab_max_size: Maximum vocabulary size, including special tokens.
        vocab_min_freq: Minimum token frequency to keep in vocabulary.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        scheduler_patience: Plateau scheduler patience.
        early_stopping_patience: Early stopping patience.
        gradient_clip: Gradient norm clip.
        embedding_dim: Embedding size for neural models.
        hidden_dim: Hidden size for recurrent/transformer blocks.
        transformer_heads: Number of attention heads.
        transformer_layers: Number of transformer blocks.
    """

    name: str
    max_tokens: int
    context_len: int
    vocab_max_size: int
    vocab_min_freq: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    scheduler_patience: int
    early_stopping_patience: int
    gradient_clip: float
    embedding_dim: int
    hidden_dim: int
    transformer_heads: int
    transformer_layers: int

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def quick_cpu_profile() -> TrainingProfile:
    """Fast profile for notebook validation and CPU fallback."""

    return TrainingProfile(
        name="quick_cpu",
        max_tokens=12_000,
        context_len=5,
        vocab_max_size=3_000,
        vocab_min_freq=2,
        batch_size=128,
        epochs=2,
        learning_rate=2e-3,
        weight_decay=1e-5,
        scheduler_patience=1,
        early_stopping_patience=2,
        gradient_clip=1.0,
        embedding_dim=128,
        hidden_dim=192,
        transformer_heads=4,
        transformer_layers=2,
    )


def full_gpu_profile() -> TrainingProfile:
    """Higher-capacity profile for GPU-first runs."""

    return TrainingProfile(
        name="full_gpu",
        max_tokens=1_500_000,
        context_len=20,
        vocab_max_size=25_000,
        vocab_min_freq=2,
        batch_size=256,
        epochs=12,
        learning_rate=1e-3,
        weight_decay=1e-5,
        scheduler_patience=2,
        early_stopping_patience=3,
        gradient_clip=1.0,
        embedding_dim=256,
        hidden_dim=384,
        transformer_heads=8,
        transformer_layers=4,
    )
