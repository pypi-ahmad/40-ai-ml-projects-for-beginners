from __future__ import annotations

import pytest
import typer

from local_rag.cli import (
    _command_guard,
    _doctor_hints,
    _exception_hints,
    compile_eval_set,
)
from local_rag.config import AppSettings
from local_rag.utils import read_jsonl, write_jsonl


def test_doctor_hints_include_actionable_ollama_and_corpus_guidance() -> None:
    settings = AppSettings()
    hints = _doctor_hints(
        failed_checks=[
            "ollama_connected",
            "embedding_model_available",
            "generation_model_available",
            "judge_model_available",
            "full_corpus_non_empty",
            "quickstart_non_empty",
        ],
        settings=settings,
    )
    joined = "\n".join(hints)
    assert "ollama serve" in joined
    assert settings.embedding_model in joined
    assert settings.generation_model in joined
    assert settings.judge_model in joined
    assert "bootstrap" in joined
    assert "build-quickstart" in joined


def test_exception_hints_cover_connection_and_collection_errors() -> None:
    settings = AppSettings()
    connection_hints = _exception_hints(
        RuntimeError("connection refused: http://127.0.0.1:11434"),
        settings,
    )
    collection_hints = _exception_hints(
        RuntimeError("collection does not exist"),
        settings,
    )
    assert any("ollama serve" in hint for hint in connection_hints)
    assert any("ingest --profile full" in hint for hint in collection_hints)


def test_command_guard_converts_runtime_errors_to_exit() -> None:
    settings = AppSettings()
    with pytest.raises(typer.Exit) as exc_info:
        with _command_guard("test-action", settings):
            raise RuntimeError("boom")
    assert exc_info.value.exit_code == 1


def test_compile_eval_set_include_unverified_builds_rows(tmp_path) -> None:
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "eval.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "query": "What is ACPI?",
                "answer_hint": "Power interface",
                "doc_id": "doc-1",
                "source_path": "docs/a.txt",
                "verified": False,
            }
        ],
    )

    compile_eval_set(
        input_path=input_path,
        output_path=output_path,
        include_unverified=True,
    )

    rows = read_jsonl(output_path)
    assert len(rows) == 1


def test_compile_eval_set_requires_verified_rows_by_default(tmp_path) -> None:
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "eval.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "query": "What is ACPI?",
                "answer_hint": "Power interface",
                "doc_id": "doc-1",
                "source_path": "docs/a.txt",
                "verified": False,
            }
        ],
    )

    with pytest.raises(typer.Exit):
        compile_eval_set(
            input_path=input_path,
            output_path=output_path,
            include_unverified=False,
        )
