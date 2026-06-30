"""Document loaders for RAG ingestion pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class DocumentLoader:
    """Load content from supported file types and URLs."""

    @staticmethod
    def load_path(path: str | Path) -> str:
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix in {".md", ".txt", ".py", ".yaml", ".yml", ".toml"}:
            return file_path.read_text(encoding="utf-8")
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if suffix == ".csv":
            with file_path.open("r", encoding="utf-8") as file:
                rows = list(csv.reader(file))
            return "\n".join(",".join(row) for row in rows)
        if suffix == ".json":
            data = json.loads(file_path.read_text(encoding="utf-8"))
            return json.dumps(data, indent=2)
        if suffix in {".html", ".htm"}:
            html = file_path.read_text(encoding="utf-8")
            import trafilatura
            from bs4 import BeautifulSoup

            extracted = trafilatura.extract(html)
            if extracted:
                return extracted
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text(separator="\n")

        return file_path.read_text(encoding="utf-8")

    @staticmethod
    def load_url(url: str) -> str:
        import httpx
        import trafilatura
        from bs4 import BeautifulSoup

        response = httpx.get(url, timeout=30)
        response.raise_for_status()
        extracted = trafilatura.extract(response.text)
        if extracted:
            return extracted
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator="\n")

    @staticmethod
    def load_dataframe_csv(path: str | Path) -> list[dict[str, Any]]:
        import pandas as pd

        frame = pd.read_csv(path)
        return frame.to_dict(orient="records")
