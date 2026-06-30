from __future__ import annotations

from pathlib import Path

from local_rag.corpus import build_quickstart_corpus, corpus_stats, validate_corpus


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_corpus_stats_counts_supported_types(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _touch(source / "a.pdf", "pdf-bytes-placeholder")
    _touch(source / "b.md", "# title")
    _touch(source / "c.txt", "plain text")
    _touch(source / "d.2", "man page text")

    stats = corpus_stats(source)
    assert stats.total_files == 4
    assert stats.pdf_files == 1
    assert stats.markdown_files == 1
    assert stats.text_files == 2
    assert stats.has_mixed_formats is True


def test_build_quickstart_corpus_creates_sample(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "quickstart"
    for idx in range(5):
        _touch(source / f"docs/manual_{idx}.pdf", "pdf")
        _touch(source / f"docs/guide_{idx}.md", "md")
        _touch(source / f"docs/page_{idx}.txt", "txt")

    sampled = build_quickstart_corpus(
        source_dir=source,
        target_dir=target,
        max_pdf=2,
        max_markdown=2,
        max_text=2,
    )
    assert sampled.pdf_files == 2
    assert sampled.markdown_files == 2
    assert sampled.text_files == 2


def test_validate_corpus_detects_missing_types(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _touch(source / "only_md.md", "md")

    validation = validate_corpus(
        source,
        min_total_files=1,
        min_pdf_files=1,
        min_markdown_files=1,
        min_text_files=1,
    )
    assert validation.ok is False
    assert any("Insufficient PDF files" in row for row in validation.errors)
