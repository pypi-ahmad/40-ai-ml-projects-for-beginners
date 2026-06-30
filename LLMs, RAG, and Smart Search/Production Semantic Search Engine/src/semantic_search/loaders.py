"""Document loading and ingestion utilities."""

from __future__ import annotations

import asyncio
import csv
import json
import random
import re
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from langdetect import LangDetectException, detect
from pypdf import PdfReader

from semantic_search.config import AppConfig
from semantic_search.logging_utils import get_logger
from semantic_search.schemas import DocumentRecord
from semantic_search.utils import hash_text, normalize_text

logger = get_logger()


class RecursiveDocumentLoader:
    """Load supported files recursively into DocumentRecord objects."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".csv", ".json", ".html", ".htm", ".docx"}

    def __init__(self, config: AppConfig):
        self.config = config

    def load(self, root_dir: str | Path, source_name: str = "filesystem") -> list[DocumentRecord]:
        """Load documents from a directory recursively."""
        root = Path(root_dir)
        documents: list[DocumentRecord] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            if self.config.security.block_hidden_files and any(part.startswith(".") for part in path.parts):
                continue
            try:
                documents.extend(self._load_file(path, source_name=source_name))
            except Exception as exc:  # noqa: BLE001
                logger.warning("file_load_failed", path=str(path), error=str(exc))
        logger.info("recursive_load_complete", count=len(documents), root=str(root))
        return self._deduplicate(documents)

    def _load_file(self, path: Path, source_name: str) -> list[DocumentRecord]:
        text_items: list[tuple[str, dict[str, str | int | float | bool]]] = []
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            text_items.append((path.read_text(encoding="utf-8", errors="ignore"), {}))
        elif suffix == ".pdf":
            reader = PdfReader(str(path))
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text_items.append((text, {"page_number": idx + 1}))
        elif suffix == ".docx":
            doc = DocxDocument(path)
            text_items.append(("\n".join(paragraph.text for paragraph in doc.paragraphs), {}))
        elif suffix == ".csv":
            rows = list(csv.DictReader(path.open("r", encoding="utf-8", errors="ignore")))
            for idx, row in enumerate(rows):
                text_items.append((json.dumps(row, ensure_ascii=False), {"row_number": idx + 1}))
        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(payload, list):
                for idx, row in enumerate(payload):
                    text_items.append((json.dumps(row, ensure_ascii=False), {"row_number": idx + 1}))
            else:
                text_items.append((json.dumps(payload, ensure_ascii=False), {}))
        elif suffix in {".html", ".htm"}:
            soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
            text_items.append((soup.get_text(separator=" ", strip=True), {}))
        else:
            return []

        docs: list[DocumentRecord] = []
        for idx, (raw_text, extra) in enumerate(text_items):
            clean_text = self._clean_text(raw_text)
            if len(clean_text) < self.config.pipeline.min_document_chars:
                continue
            doc_hash = hash_text(clean_text)
            metadata = {
                "source": source_name,
                "filename": path.name,
                "extension": suffix,
                "path": str(path),
                **extra,
            }
            docs.append(
                DocumentRecord(
                    doc_id=f"{source_name}-{path.stem}-{idx}-{doc_hash[:10]}",
                    source=source_name,
                    filename=path.name,
                    text=clean_text[: self.config.pipeline.max_document_chars],
                    language=self._detect_language(clean_text),
                    document_hash=doc_hash,
                    metadata=metadata,
                )
            )
        return docs

    def _clean_text(self, text: str) -> str:
        clean = normalize_text(text)
        if self.config.pipeline.normalize_whitespace:
            clean = re.sub(r"\s+", " ", clean)
        return clean

    def _detect_language(self, text: str) -> str | None:
        if not self.config.pipeline.detect_language:
            return None
        try:
            return detect(text[:1000])
        except LangDetectException:
            return None

    def _deduplicate(self, documents: list[DocumentRecord]) -> list[DocumentRecord]:
        if not self.config.pipeline.deduplicate:
            return documents
        seen: set[str] = set()
        unique: list[DocumentRecord] = []
        for doc in documents:
            if doc.document_hash in seen:
                continue
            seen.add(doc.document_hash)
            unique.append(doc)
        return unique


def _is_allowed_domain(url: str, allowed_domains: Iterable[str]) -> bool:
    domain = urlparse(url).netloc.lower().strip()
    if not domain:
        return False
    if not allowed_domains:
        return True
    return domain in {d.lower() for d in allowed_domains}


async def _fetch_article_body(
    session: aiohttp.ClientSession,
    url: str,
    timeout_seconds: int,
    max_chars: int,
) -> str | None:
    """Fetch and parse article body from URL."""
    try:
        async with session.get(url, timeout=timeout_seconds) as response:
            if response.status != 200:
                return None
            html = await response.text()
            soup = BeautifulSoup(html, "html.parser")
            article = soup.find("article")
            if article:
                text = article.get_text(separator=" ", strip=True)
            else:
                paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
                text = " ".join(paragraphs)
            text = normalize_text(text)
            return text[:max_chars] if text else None
    except Exception:  # noqa: BLE001
        return None


async def enrich_documents_with_urls(documents: list[DocumentRecord], config: AppConfig) -> list[DocumentRecord]:
    """Enrich documents using source URLs with bounded async requests."""
    if not config.dataset.enable_article_enrichment:
        return documents

    semaphore = asyncio.Semaphore(config.dataset.max_concurrent_enrichment)
    max_enrich = max(0, int(config.dataset.enrichment_sample_size))
    enrich_budget = {"count": 0}

    async def enrich_one(doc: DocumentRecord, session: aiohttp.ClientSession) -> DocumentRecord:
        url = doc.url or doc.metadata.get("link")
        if not url or not _is_allowed_domain(str(url), config.dataset.allowed_url_domains):
            return doc
        if enrich_budget["count"] >= max_enrich:
            return doc

        async with semaphore:
            enrich_budget["count"] += 1
            article_text = await _fetch_article_body(
                session=session,
                url=str(url),
                timeout_seconds=config.dataset.request_timeout_seconds,
                max_chars=config.dataset.max_article_chars,
            )

        if not article_text:
            return doc

        merged_text = normalize_text(f"{doc.title or ''}. {article_text}")
        updated = doc.model_copy(update={
            "text": merged_text[: config.pipeline.max_document_chars],
            "document_hash": hash_text(merged_text),
            "version": doc.version + 1,
            "metadata": {
                **doc.metadata,
                "enriched": True,
                "enrichment_source": "url_fetch",
            },
        })
        return updated

    timeout = aiohttp.ClientTimeout(total=config.dataset.request_timeout_seconds + 2)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        enriched = await asyncio.gather(*(enrich_one(doc, session) for doc in documents))
    logger.info(
        "url_enrichment_complete",
        attempted=len(documents),
        budget=max_enrich,
        consumed=enrich_budget["count"],
    )
    return enriched


def load_huggingface_huffpost(config: AppConfig) -> list[DocumentRecord]:
    """Load and normalize documents from Hugging Face dataset."""
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install `datasets` package to use HF ingestion") from exc

    dataset_candidates = [
        config.dataset.hf_dataset_id,
        *config.dataset.hf_fallback_dataset_ids,
    ]
    split = config.dataset.split
    ds = None
    used_dataset_id = None
    errors: list[str] = []

    for dataset_id in dataset_candidates:
        try:
            ds = load_dataset(dataset_id, split=split)
            used_dataset_id = dataset_id
            break
        except RuntimeError as exc:
            if "Dataset scripts are no longer supported" not in str(exc):
                errors.append(f"{dataset_id}: {exc}")
                continue

            api = HfApi()
            repo_files = api.list_repo_files(repo_id=dataset_id, repo_type="dataset")
            parquet_files = [path for path in repo_files if path.endswith(".parquet")]
            if not parquet_files:
                errors.append(f"{dataset_id}: script-only and no parquet fallback")
                continue

            split_paths = [
                path
                for path in parquet_files
                if f"/{split}" in path or path.startswith(f"{split}/") or split in path
            ]
            selected_paths = split_paths or parquet_files
            hf_data_files = [f"hf://datasets/{dataset_id}/{path}" for path in selected_paths]
            ds = load_dataset("parquet", data_files=hf_data_files, split="train")
            used_dataset_id = dataset_id
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{dataset_id}: {exc}")
            continue

    if ds is None or used_dataset_id is None:
        raise RuntimeError(
            "No Hugging Face dataset candidate could be loaded. "
            f"Tried: {dataset_candidates}. Errors: {errors}"
        )

    total_rows = len(ds)
    sample_size = min(config.dataset.sample_size, total_rows)
    rng = random.Random(config.dataset.seed)
    sampled_indices = sorted(rng.sample(range(total_rows), sample_size))

    docs: list[DocumentRecord] = []
    for idx in sampled_indices:
        row = ds[idx]
        title = normalize_text(str(row.get("headline") or ""))
        desc = normalize_text(str(row.get("short_description") or ""))
        text = normalize_text(f"{title}. {desc}")
        if len(text) < config.pipeline.min_document_chars:
            continue

        date_value = str(row.get("date") or "")
        url_value = str(row.get("link") or "")
        author = str(row.get("authors") or "").strip() or None
        category = str(row.get("category") or "").strip() or None

        doc_hash = hash_text(text)
        docs.append(
            DocumentRecord(
                doc_id=f"hf-huffpost-{idx}-{doc_hash[:10]}",
                source=config.dataset.hf_dataset_id,
                title=title or None,
                text=text[: config.pipeline.max_document_chars],
                category=category,
                author=author,
                published_date=date_value or None,
                language="en",
                url=url_value or None,
                document_hash=doc_hash,
                metadata={
                    "row_index": idx,
                    "dataset": used_dataset_id,
                    "split": config.dataset.split,
                    "source": "huggingface",
                    "filename": "huffpost_row",
                },
            )
        )

    logger.info(
        "hf_load_complete",
        dataset=used_dataset_id,
        sampled=sample_size,
        loaded=len(docs),
    )
    return docs


def write_documents_jsonl(documents: list[DocumentRecord], output_path: str | Path) -> None:
    """Write canonical documents to JSONL."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(doc.model_dump_json() + "\n")
    logger.info("documents_written", count=len(documents), path=str(target))


def read_documents_jsonl(path: str | Path) -> list[DocumentRecord]:
    """Read canonical document records from JSONL."""
    docs: list[DocumentRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            docs.append(DocumentRecord.model_validate_json(line))
    return docs


def export_multiformat_subset(documents: list[DocumentRecord], export_dir: str | Path, limit: int = 200) -> None:
    """Export a subset to multiple formats to validate recursive loaders."""
    target = Path(export_dir)
    target.mkdir(parents=True, exist_ok=True)
    subset = documents[:limit]

    txt_dir = target / "txt"
    md_dir = target / "md"
    html_dir = target / "html"
    json_dir = target / "json"
    csv_dir = target / "csv"
    for d in (txt_dir, md_dir, html_dir, json_dir, csv_dir):
        d.mkdir(parents=True, exist_ok=True)

    csv_rows: list[dict[str, str]] = []
    for idx, doc in enumerate(subset):
        text = doc.text
        (txt_dir / f"doc_{idx}.txt").write_text(text, encoding="utf-8")
        (md_dir / f"doc_{idx}.md").write_text(f"# {doc.title or 'Untitled'}\n\n{text}", encoding="utf-8")
        (html_dir / f"doc_{idx}.html").write_text(
            f"<html><body><h1>{doc.title or 'Untitled'}</h1><p>{text}</p></body></html>",
            encoding="utf-8",
        )
        (json_dir / f"doc_{idx}.json").write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        csv_rows.append(
            {
                "doc_id": doc.doc_id,
                "title": doc.title or "",
                "text": doc.text,
                "category": doc.category or "",
                "author": doc.author or "",
                "date": doc.published_date or "",
            }
        )

    pd.DataFrame(csv_rows).to_csv(csv_dir / "documents.csv", index=False)
    logger.info("multiformat_subset_exported", count=len(subset), path=str(target))
