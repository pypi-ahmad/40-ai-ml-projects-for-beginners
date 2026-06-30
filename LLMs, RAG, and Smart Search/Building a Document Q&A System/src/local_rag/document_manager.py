"""Document lifecycle management for upload, delete, update, and catalog reporting."""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from local_rag.loaders import DocumentLoader
from local_rag.utils import json_dump, sha256_file, sha256_text


@dataclass(slots=True)
class ManagedDocument:
    """Document catalog entry used by UI and reports."""

    source_path: str
    document_name: str
    file_suffix: str
    file_hash: str
    version_id: str
    file_size_bytes: int
    last_modified_ns: int


class DocumentManager:
    """Manage local document files and metadata catalog."""

    def __init__(self, base_dir: Path, catalog_path: Path) -> None:
        self.base_dir = base_dir
        self.catalog_path = catalog_path
        self.loader = DocumentLoader(base_dir=base_dir)

    def list_documents(self) -> list[ManagedDocument]:
        """List all supported documents with version metadata."""

        docs: list[ManagedDocument] = []
        for path in sorted(self.loader._iter_supported_files(self.base_dir)):
            source_path = self._relative(path)
            file_hash = sha256_file(path)
            version_id = sha256_text(f"{file_hash}::{path.stat().st_mtime_ns}")
            docs.append(
                ManagedDocument(
                    source_path=source_path,
                    document_name=path.name,
                    file_suffix=path.suffix.lower(),
                    file_hash=file_hash,
                    version_id=version_id,
                    file_size_bytes=path.stat().st_size,
                    last_modified_ns=path.stat().st_mtime_ns,
                )
            )
        return docs

    def delete_document(self, source_path: str) -> bool:
        """Delete document by source-relative path."""

        target = self.base_dir / source_path
        if not target.exists() or not target.is_file():
            return False
        target.unlink()
        return True

    def upsert_document(self, source_path: str, payload: bytes) -> Path:
        """Insert or replace a document file."""

        target = self.base_dir / source_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target

    def import_from_folder(self, folder: Path, *, move: bool = False) -> int:
        """Ingest supported files from external folder into managed directory."""

        count = 0
        for source in self.loader._iter_supported_files(folder):
            rel = source.name
            target = self.base_dir / "imports" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if move:
                shutil.move(str(source), str(target))
            else:
                shutil.copy2(source, target)
            count += 1
        return count

    def write_catalog(self) -> Path:
        """Persist current catalog to disk."""

        rows = [asdict(row) for row in self.list_documents()]
        json_dump(self.catalog_path, {"rows": rows, "count": len(rows)})
        return self.catalog_path

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.base_dir))
        except ValueError:
            return str(path)
