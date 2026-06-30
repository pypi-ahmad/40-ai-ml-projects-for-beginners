"""Train baseline + neural language models and generate benchmark tables.

Usage:
    uv run python scripts/train_and_benchmark.py --profile quick
    uv run python scripts/train_and_benchmark.py --profile full --prefer-gpu
    uv run python scripts/train_and_benchmark.py --profile quick --models LSTM,Transformer
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import pandas as pd

from utils.config import (
    PathConfig,
    TrainingProfile,
    full_gpu_profile,
    quick_cpu_profile,
)
from utils.models import (
    CNN_LSTM_LM,
    GRU_LM,
    LSTM_LM,
    BigramModel,
    BiLSTM_LM,
    MostFrequentWordModel,
    StackedLSTM_LM,
    TransformerLM,
    TrigramModel,
    UnigramModel,
)
from utils.pipeline import (
    build_corpus_and_profile,
    prepare_tokenized_artifacts,
    resolve_device,
    run_ngram_benchmarks,
    save_prepared_bundle,
    train_neural_model,
    write_json,
)
from utils.reproducibility import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train predictive keyboard models.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root path.",
    )
    parser.add_argument(
        "--profile",
        choices=["quick", "full"],
        default="quick",
        help="Training profile.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefer-gpu", action="store_true")
    parser.add_argument("--include-wikitext", action="store_true")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated neural model names to train, or 'all'.",
    )
    parser.add_argument("--wikitext-train-tokens", type=int, default=1_500_000)
    parser.add_argument("--wikitext-val-tokens", type=int, default=200_000)
    parser.add_argument("--wikitext-test-tokens", type=int, default=200_000)
    return parser.parse_args()


def build_profile(profile_name: str) -> TrainingProfile:
    if profile_name == "full":
        return full_gpu_profile()
    return quick_cpu_profile()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    project_root = args.project_root.resolve()
    paths = PathConfig.from_project_root(project_root)
    paths.ensure_dirs()

    profile = build_profile(args.profile)
    device = resolve_device(prefer_gpu=args.prefer_gpu)

    bundle, dataset_stats = build_corpus_and_profile(
        project_root=project_root,
        include_wikitext=args.include_wikitext,
        wikitext_train_tokens=args.wikitext_train_tokens,
        wikitext_val_tokens=args.wikitext_val_tokens,
        wikitext_test_tokens=args.wikitext_test_tokens,
    )
    save_prepared_bundle(project_root, bundle)

    artifacts = prepare_tokenized_artifacts(bundle=bundle, profile=profile, seed=args.seed)
    vocab_path = paths.results_dir / f"vocab_{profile.name}.json"
    artifacts.vocabulary.save(vocab_path)

    # -------------------------
    # N-gram baselines
    # -------------------------
    ngram_models = {
        "MostFrequent": MostFrequentWordModel(vocab_size=len(artifacts.vocabulary)),
        "Unigram": UnigramModel(vocab_size=len(artifacts.vocabulary), smoothing=1.0),
        "Bigram": BigramModel(vocab_size=len(artifacts.vocabulary), smoothing=1.0),
        "Trigram": TrigramModel(vocab_size=len(artifacts.vocabulary), smoothing=1.0),
    }
    ngram_df = run_ngram_benchmarks(
        models=ngram_models,
        train_ids=artifacts.train_ids,
        test_loader=artifacts.dataloaders["test"],
        device="cpu",
    )

    # -------------------------
    # Neural language models
    # -------------------------
    neural_factories = {
        "LSTM": lambda: LSTM_LM(
            vocab_size=len(artifacts.vocabulary),
            embedding_dim=profile.embedding_dim,
            hidden_dim=profile.hidden_dim,
            num_layers=1,
        ),
        "StackedLSTM": lambda: StackedLSTM_LM(
            vocab_size=len(artifacts.vocabulary),
            embedding_dim=profile.embedding_dim,
            hidden_dim=profile.hidden_dim,
            num_layers=2,
        ),
        "BiLSTM": lambda: BiLSTM_LM(
            vocab_size=len(artifacts.vocabulary),
            embedding_dim=profile.embedding_dim,
            hidden_dim=max(profile.hidden_dim // 2, 64),
            num_layers=2,
        ),
        "GRU": lambda: GRU_LM(
            vocab_size=len(artifacts.vocabulary),
            embedding_dim=profile.embedding_dim,
            hidden_dim=profile.hidden_dim,
            num_layers=2,
        ),
        "CNN_LSTM": lambda: CNN_LSTM_LM(
            vocab_size=len(artifacts.vocabulary),
            embedding_dim=profile.embedding_dim,
            hidden_dim=profile.hidden_dim,
            num_filters=max(profile.embedding_dim // 2, 64),
        ),
        "Transformer": lambda: TransformerLM(
            vocab_size=len(artifacts.vocabulary),
            embedding_dim=profile.embedding_dim,
            hidden_dim=profile.hidden_dim,
            nhead=profile.transformer_heads,
            num_layers=profile.transformer_layers,
        ),
    }
    requested_models = [item.strip() for item in args.models.split(",") if item.strip()]
    if requested_models and requested_models != ["all"]:
        unknown = sorted(set(requested_models) - set(neural_factories))
        if unknown:
            raise ValueError(f"Unknown model(s) in --models: {unknown}")
        neural_factories = {name: neural_factories[name] for name in requested_models}

    neural_rows: list[dict[str, float | str]] = []
    model_registry: dict[str, dict[str, object]] = {}
    run_suffix = str(int(time.time()))
    for model_name, factory in neural_factories.items():
        model = factory()
        start = time.perf_counter()
        metrics, benchmark = train_neural_model(
            model_name=model_name,
            model=model,
            artifacts=artifacts,
            profile=profile,
            output_dir=paths.outputs_dir,
            device=device,
            run_suffix=run_suffix,
        )
        train_seconds = time.perf_counter() - start

        row: dict[str, float | str] = {
            "model": model_name,
            "family": "neural",
            "train_time_sec": float(train_seconds),
        }
        row.update(metrics)
        row.update(benchmark)
        neural_rows.append(row)

        model_registry[model_name] = {
            "checkpoint_path": str(
                paths.checkpoints_dir / f"{model_name}_{profile.name}_{run_suffix}.pt"
            ),
            "vocab_path": str(vocab_path),
            "profile_name": profile.name,
            "context_len": profile.context_len,
            "embedding_dim": profile.embedding_dim,
            "hidden_dim": profile.hidden_dim,
            "transformer_heads": profile.transformer_heads,
            "transformer_layers": profile.transformer_layers,
            "vocab_size": len(artifacts.vocabulary),
        }

    neural_df = pd.DataFrame(neural_rows)

    leaderboard = pd.concat([ngram_df, neural_df], ignore_index=True)
    leaderboard["rank_score"] = (
        leaderboard["top5_accuracy"] * 100
        - leaderboard["perplexity"].clip(lower=1).map(lambda x: float(x)).apply(lambda x: min(x, 500)) * 0.05
    )
    leaderboard = leaderboard.sort_values(
        by=["rank_score", "top5_accuracy", "perplexity"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    leaderboard_csv = paths.results_dir / f"leaderboard_{profile.name}.csv"
    leaderboard_json = paths.results_dir / f"leaderboard_{profile.name}.json"
    leaderboard.to_csv(leaderboard_csv, index=False)
    leaderboard_json.write_text(
        leaderboard.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    split_hashes = {
        "train_text_sha256": hashlib.sha256(bundle.train_text.encode("utf-8")).hexdigest(),
        "val_text_sha256": hashlib.sha256(bundle.val_text.encode("utf-8")).hexdigest(),
        "test_text_sha256": hashlib.sha256(bundle.test_text.encode("utf-8")).hexdigest(),
        "vocab_sha256": hashlib.sha256(
            vocab_path.read_bytes()
        ).hexdigest()
        if vocab_path.exists()
        else "",
    }

    run_meta = {
        "profile": profile.to_dict(),
        "run_suffix": run_suffix,
        "device": device,
        "seed": args.seed,
        "vocab_size": len(artifacts.vocabulary),
        "dataset_stats": dataset_stats,
        "split_hashes": split_hashes,
        "artifacts": {
            "vocab": str(vocab_path),
            "leaderboard_csv": str(leaderboard_csv),
            "leaderboard_json": str(leaderboard_json),
        },
    }
    write_json(paths.results_dir / f"run_metadata_{profile.name}.json", run_meta)
    write_json(paths.results_dir / f"model_registry_{profile.name}.json", model_registry)

    print("Training and benchmarking completed.")
    print(f"Device: {device}")
    print(f"Vocab size: {len(artifacts.vocabulary)}")
    print(f"Leaderboard: {leaderboard_csv}")


if __name__ == "__main__":
    main()
