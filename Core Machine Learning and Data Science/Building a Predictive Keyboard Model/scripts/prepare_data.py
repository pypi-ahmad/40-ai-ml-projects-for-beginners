"""Prepare Sherlock + optional WikiText corpus for predictive keyboard project.

Usage:
    uv run python scripts/prepare_data.py --include-wikitext
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.config import PathConfig
from utils.data import validate_corpus_bundle
from utils.pipeline import build_corpus_and_profile, save_prepared_bundle, write_json
from utils.reproducibility import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare predictive keyboard corpus.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root path.",
    )
    parser.add_argument(
        "--include-wikitext",
        action="store_true",
        help="Include WikiText-103 subset in addition to Sherlock corpus.",
    )
    parser.add_argument("--wikitext-train-tokens", type=int, default=1_500_000)
    parser.add_argument("--wikitext-val-tokens", type=int, default=200_000)
    parser.add_argument("--wikitext-test-tokens", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    project_root = args.project_root.resolve()
    paths = PathConfig.from_project_root(project_root)
    paths.ensure_dirs()

    bundle, stats = build_corpus_and_profile(
        project_root=project_root,
        include_wikitext=args.include_wikitext,
        wikitext_train_tokens=args.wikitext_train_tokens,
        wikitext_val_tokens=args.wikitext_val_tokens,
        wikitext_test_tokens=args.wikitext_test_tokens,
    )

    save_prepared_bundle(project_root, bundle)

    profile_payload = {
        "metadata": bundle.metadata,
        "train": stats["train"],
        "val": stats["val"],
        "test": stats["test"],
    }
    validation_payload = validate_corpus_bundle(bundle)
    raw_dir = paths.data_dir / "raw"
    discovered_files = [
        str(path.relative_to(project_root))
        for path in sorted(raw_dir.rglob("*"))
        if path.is_file()
    ]
    validation_payload["file_discovery"] = {
        "raw_dir": str(raw_dir),
        "num_files": len(discovered_files),
        "files": discovered_files,
    }

    write_json(paths.results_dir / "dataset_profile.json", profile_payload)
    write_json(paths.results_dir / "dataset_validation.json", validation_payload)
    (paths.results_dir / "dataset_profile.md").write_text(
        "# Dataset Profile\n\n" + json.dumps(profile_payload, indent=2),
        encoding="utf-8",
    )
    (paths.results_dir / "dataset_validation.md").write_text(
        "# Dataset Validation\n\n" + json.dumps(validation_payload, indent=2),
        encoding="utf-8",
    )

    print("Data prepared successfully.")
    print(f"Saved: {paths.data_dir / 'processed'}")
    print(f"Profile: {paths.results_dir / 'dataset_profile.json'}")
    print(f"Validation: {paths.results_dir / 'dataset_validation.json'}")


if __name__ == "__main__":
    main()
