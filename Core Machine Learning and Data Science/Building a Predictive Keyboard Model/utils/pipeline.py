"""End-to-end helpers for predictive keyboard training pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from .benchmarking import benchmark_model_inference
from .config import PathConfig, TrainingProfile
from .data import (
    CorpusBundle,
    corpus_statistics,
    prepare_combined_corpus,
    save_corpus_bundle,
)
from .evaluation import dataloader_metrics
from .sequence_builder import build_dataloaders_from_ids
from .tokenization import NLTKTokenizerBackend, normalize_text
from .trainer import fit_model, load_checkpoint
from .vocabulary import Vocabulary


@dataclass(slots=True)
class PreparedArtifacts:
    """Prepared dataset artifacts used by model training."""

    vocabulary: Vocabulary
    train_ids: list[int]
    val_ids: list[int]
    test_ids: list[int]
    dataloaders: dict[str, Any]


def resolve_device(prefer_gpu: bool = True) -> str:
    """Resolve run device with GPU-first fallback logic."""

    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_corpus_and_profile(
    project_root: Path,
    *,
    include_wikitext: bool,
    wikitext_train_tokens: int,
    wikitext_val_tokens: int,
    wikitext_test_tokens: int,
) -> tuple[CorpusBundle, dict[str, dict[str, int | float]]]:
    """Prepare combined corpus and profile each split."""

    bundle = prepare_combined_corpus(
        project_root=project_root,
        include_wikitext=include_wikitext,
        wikitext_train_tokens=wikitext_train_tokens,
        wikitext_val_tokens=wikitext_val_tokens,
        wikitext_test_tokens=wikitext_test_tokens,
    )
    stats = {
        "train": corpus_statistics(bundle.train_text),
        "val": corpus_statistics(bundle.val_text),
        "test": corpus_statistics(bundle.test_text),
    }
    return bundle, stats


def prepare_tokenized_artifacts(
    bundle: CorpusBundle,
    profile: TrainingProfile,
    *,
    seed: int = 42,
) -> PreparedArtifacts:
    """Tokenize splits, build vocabulary, and create dataloaders."""

    tokenizer = NLTKTokenizerBackend()
    train_tokens = tokenizer.tokenize(normalize_text(bundle.train_text))
    val_tokens = tokenizer.tokenize(normalize_text(bundle.val_text))
    test_tokens = tokenizer.tokenize(normalize_text(bundle.test_text))

    if profile.max_tokens > 0:
        train_tokens = train_tokens[: profile.max_tokens]
        val_tokens = val_tokens[: max(1, int(profile.max_tokens * 0.1))]
        test_tokens = test_tokens[: max(1, int(profile.max_tokens * 0.1))]

    vocab = Vocabulary(min_freq=profile.vocab_min_freq, max_size=profile.vocab_max_size)
    vocab.build([train_tokens])

    train_ids = vocab.encode_with_special(train_tokens)
    val_ids = vocab.encode_with_special(val_tokens)
    test_ids = vocab.encode_with_special(test_tokens)

    train_loader = build_dataloaders_from_ids(
        ids=train_ids,
        context_len=profile.context_len,
        batch_size=profile.batch_size,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=seed,
    )["train"]

    # Validation and test dataloaders built from their own splits for clean eval.
    val_loader = build_dataloaders_from_ids(
        ids=val_ids,
        context_len=profile.context_len,
        batch_size=profile.batch_size,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=seed,
    )["train"]
    test_loader = build_dataloaders_from_ids(
        ids=test_ids,
        context_len=profile.context_len,
        batch_size=profile.batch_size,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=seed,
    )["train"]

    if len(val_loader.dataset) == 0:  # type: ignore[arg-type]
        raise ValueError("Validation split produced zero sequences. Increase token budget.")
    if len(test_loader.dataset) == 0:  # type: ignore[arg-type]
        raise ValueError("Test split produced zero sequences. Increase token budget.")

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    return PreparedArtifacts(
        vocabulary=vocab,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        dataloaders=dataloaders,
    )


def _evaluate_ngram_model(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: str,
    max_batches: int = 20,
) -> dict[str, float]:
    limited_batches = islice(dataloader, max_batches)
    return dataloader_metrics(
        model=model,
        dataloader=limited_batches,
        criterion=criterion,
        device=device,
    )


def train_neural_model(
    *,
    model_name: str,
    model: nn.Module,
    artifacts: PreparedArtifacts,
    profile: TrainingProfile,
    output_dir: Path,
    device: str,
    run_suffix: str = "",
) -> tuple[dict[str, float], dict[str, float]]:
    """Train one neural model and return evaluation + benchmark metrics."""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=profile.learning_rate,
        weight_decay=profile.weight_decay,
    )

    model.to(device)

    suffix = f"_{run_suffix}" if run_suffix else ""
    checkpoint_path = output_dir / "checkpoints" / f"{model_name}_{profile.name}{suffix}.pt"
    history_path = output_dir / "results" / f"{model_name}_{profile.name}{suffix}_history.json"

    fit_model(
        model=model,
        train_loader=artifacts.dataloaders["train"],
        val_loader=artifacts.dataloaders["val"],
        optimizer=optimizer,
        criterion=criterion,
        epochs=profile.epochs,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        clip=profile.gradient_clip,
        scheduler_patience=profile.scheduler_patience,
        early_stopping_patience=profile.early_stopping_patience,
        device=device,
        use_amp=device.startswith("cuda"),
    )
    if checkpoint_path.exists():
        load_checkpoint(model, path=checkpoint_path, device=device)

    metrics = dataloader_metrics(
        model=model,
        dataloader=artifacts.dataloaders["test"],
        criterion=criterion,
        device=device,
    )
    benchmark = benchmark_model_inference(
        model=model,
        dataloader=artifacts.dataloaders["test"],
        device=device,
        max_batches=30,
    )
    return metrics, benchmark


def run_ngram_benchmarks(
    *,
    models: dict[str, nn.Module],
    train_ids: list[int],
    test_loader,
    device: str,
) -> pd.DataFrame:
    """Fit and evaluate n-gram baselines."""

    criterion = nn.NLLLoss()
    rows: list[dict[str, float | str]] = []

    for name, model in tqdm(models.items(), desc="N-gram baselines"):
        fit_method = getattr(model, "fit", None)
        if callable(fit_method):
            fit_method(train_ids)

        metrics = _evaluate_ngram_model(model, test_loader, criterion, device, max_batches=30)
        bench = benchmark_model_inference(model, test_loader, device=device, max_batches=30)
        row = {"model": name, "family": "ngram"}
        row.update(metrics)
        row.update(bench)
        rows.append(row)

    return pd.DataFrame(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON file with parent directory creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_prepared_bundle(project_root: Path, bundle: CorpusBundle) -> None:
    """Persist train/val/test text bundle under data/processed."""

    paths = PathConfig.from_project_root(project_root)
    save_corpus_bundle(bundle, paths.data_dir / "processed")
