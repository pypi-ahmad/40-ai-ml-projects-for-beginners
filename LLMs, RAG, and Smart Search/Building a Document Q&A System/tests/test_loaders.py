from __future__ import annotations

from pathlib import Path

from pypdf import PdfWriter

from local_rag.loaders import DocumentLoader


def test_load_text_and_markdown(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)

    txt = docs_dir / "a.txt"
    txt.write_text("alpha beta gamma", encoding="utf-8")

    md = docs_dir / "b.md"
    md.write_text("# Header\n`code` and [link](https://example.com)", encoding="utf-8")

    loader = DocumentLoader(base_dir=docs_dir)
    loaded = loader.load_directory()

    paths = {row.metadata["source_path"] for row in loaded}
    assert "a.txt" in paths
    assert "b.md" in paths


def test_load_pdf_graceful_on_blank_page(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)

    pdf_path = docs_dir / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    with pdf_path.open("wb") as handle:
        writer.write(handle)

    loader = DocumentLoader(base_dir=docs_dir)
    loaded = loader.load_file(pdf_path)

    # Blank PDF should not crash; may produce zero extracted pages.
    assert isinstance(loaded, list)
