"""Command-line interface for llmft workflows."""

from __future__ import annotations

import argparse
import json
import logging

from llmft.inference.backends import InferenceRouter
from llmft.pipeline import ProjectRunner
from llmft.serving.api import create_app
from llmft.ui.app import run_streamlit_app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmft", description="LLM fine-tuning framework CLI")
    parser.add_argument("--config", default="configs/project.yaml", help="Path to YAML config")

    sub = parser.add_subparsers(dest="group", required=True)

    env = sub.add_parser("env")
    env_sub = env.add_subparsers(dest="action", required=True)
    env_sub.add_parser("validate")

    data = sub.add_parser("data")
    data_sub = data.add_subparsers(dest="action", required=True)
    data_sub.add_parser("build")

    train = sub.add_parser("train")
    train_sub = train.add_subparsers(dest="action", required=True)
    train_sft = train_sub.add_parser("sft")
    train_sft.add_argument("--dry-run", action="store_true")
    train_sub.add_parser("hpo")

    evaluate = sub.add_parser("eval")
    eval_sub = evaluate.add_subparsers(dest="action", required=True)
    eval_sub.add_parser("run")

    bench = sub.add_parser("bench")
    bench_sub = bench.add_subparsers(dest="action", required=True)
    bench_sub.add_parser("run")

    infer = sub.add_parser("infer")
    infer_sub = infer.add_subparsers(dest="action", required=True)
    infer_run = infer_sub.add_parser("run")
    infer_run.add_argument("--prompt", default="Explain LoRA in one sentence.")
    infer_run.add_argument("--backend", default=None)

    export = sub.add_parser("export")
    export_sub = export.add_subparsers(dest="action", required=True)
    export_sub.add_parser("run")

    serve = sub.add_parser("serve")
    serve_sub = serve.add_subparsers(dest="action", required=True)
    serve_sub.add_parser("api")

    ui = sub.add_parser("ui")
    ui_sub = ui.add_subparsers(dest="action", required=True)
    ui_sub.add_parser("streamlit")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Execute CLI command."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    logger = logging.getLogger("llmft.cli")

    runner = ProjectRunner.from_config_path(args.config)

    if args.group == "env" and args.action == "validate":
        path = runner.validate_env()
        logger.info("env report: %s", path)
        return

    if args.group == "data" and args.action == "build":
        bundle = runner.build_data()
        logger.info("dataset manifest: %s", bundle.manifest_path)
        return

    if args.group == "train" and args.action == "sft":
        report = runner.train_sft(dry_run=args.dry_run)
        logger.info("training report: %s", json.dumps({"run_id": report.run_id, "eval_loss": report.eval_loss}))
        return

    if args.group == "train" and args.action == "hpo":
        report = runner.train_sft(dry_run=False)
        logger.info("hpo stub from run: %s", report.run_id)
        return

    if args.group == "eval" and args.action == "run":
        path = runner.run_evaluation()
        logger.info("evaluation report: %s", path)
        return

    if args.group == "bench" and args.action == "run":
        path = runner.run_benchmark()
        logger.info("benchmark report: %s", path)
        return

    if args.group == "infer" and args.action == "run":
        if args.backend:
            runner.config.inference.backend = args.backend
        path = runner.run_inference(args.prompt)
        logger.info("inference report: %s", path)
        return

    if args.group == "export" and args.action == "run":
        path = runner.run_export()
        logger.info("export summary: %s", path)
        return

    if args.group == "serve" and args.action == "api":
        router = InferenceRouter(runner.config.inference)
        app = create_app(router)
        try:
            import uvicorn  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("uvicorn not installed. Install extra: llmft-framework[serve]") from exc
        uvicorn.run(app, host=runner.config.serve.host, port=runner.config.serve.port)
        return

    if args.group == "ui" and args.action == "streamlit":
        run_streamlit_app()
        return

    parser.error("Unknown command")


if __name__ == "__main__":
    main()
