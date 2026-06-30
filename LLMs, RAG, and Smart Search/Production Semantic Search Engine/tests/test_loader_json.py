from pathlib import Path

from semantic_search.config import load_config
from semantic_search.loaders import RecursiveDocumentLoader


def test_recursive_loader_reads_json_rows(tmp_path: Path):
    data = '[{"title": "A", "text": "hello world"}, {"title": "B", "text": "another text"}]'
    path = tmp_path / "sample.json"
    path.write_text(data, encoding="utf-8")

    cfg = load_config("config/default.yaml")
    loader = RecursiveDocumentLoader(cfg)
    docs = loader.load(tmp_path, source_name="test")

    assert len(docs) == 2
    assert all(doc.text for doc in docs)
