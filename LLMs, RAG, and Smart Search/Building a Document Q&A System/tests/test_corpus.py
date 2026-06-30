from __future__ import annotations

from pathlib import Path

from local_rag.corpus import build_quickstart_corpus, corpus_stats, ensure_mixed_formats


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


def test_ensure_mixed_formats_backfills_missing_types(tmp_path: Path) -> None:
    source = tmp_path / "source"
    fallback = tmp_path / "fallback"
    _touch(source / "only_md.md", "md")
    _touch(fallback / "seed.pdf", "pdf")
    _touch(fallback / "seed.txt", "txt")
    _touch(fallback / "seed.md", "md")

    stats = ensure_mixed_formats(
        source_dir=source,
        fallback_dir=fallback,
        copies_per_missing_type=1,
    )
    assert stats.has_mixed_formats is True
    assert stats.pdf_files >= 1
    assert stats.text_files >= 1
