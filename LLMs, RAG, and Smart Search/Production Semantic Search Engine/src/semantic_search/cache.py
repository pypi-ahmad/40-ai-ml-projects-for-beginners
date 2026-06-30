"""SQLite-backed caches for embeddings and query results."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from semantic_search.utils import hash_text


class EmbeddingCache:
    """Cache vectors by model + input hash."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    model_name TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    PRIMARY KEY (model_name, text_hash)
                )
                """
            )

    def get(self, model_name: str, text: str) -> np.ndarray | None:
        text_hash = hash_text(text)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT vector FROM embeddings WHERE model_name = ? AND text_hash = ?",
                (model_name, text_hash),
            ).fetchone()
        if not row:
            return None
        return np.asarray(json.loads(row[0]), dtype=np.float32)

    def set(self, model_name: str, text: str, vector: np.ndarray) -> None:
        text_hash = hash_text(text)
        payload = json.dumps(vector.tolist())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings(model_name, text_hash, vector) VALUES (?, ?, ?)",
                (model_name, text_hash, payload),
            )

    def set_many(self, model_name: str, rows: list[tuple[str, np.ndarray]]) -> None:
        """Batch upsert embeddings for significant ingestion speedup."""
        if not rows:
            return
        payload = [
            (
                model_name,
                hash_text(text),
                json.dumps(vector.tolist()),
            )
            for text, vector in rows
        ]
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings(model_name, text_hash, vector) VALUES (?, ?, ?)",
                payload,
            )


class QueryCache:
    """Cache search responses for identical request signatures."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_cache (
                    cache_key TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL
                )
                """
            )

    def make_key(
        self,
        query: str,
        mode: str,
        model_name: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> str:
        payload = json.dumps(
            {
                "query": query,
                "mode": mode,
                "model": model_name,
                "top_k": top_k,
                "filters": filters or {},
            },
            sort_keys=True,
        )
        return hash_text(payload)

    def get(self, key: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT response_json FROM query_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def set(self, key: str, payload: dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO query_cache(cache_key, response_json) VALUES (?, ?)",
                (key, json.dumps(payload)),
            )


class SemanticQueryCache:
    """Cache responses by query embedding similarity."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_query_cache (
                    cache_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    filters_json TEXT NOT NULL,
                    query_vector TEXT NOT NULL,
                    response_json TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def get_similar(
        self,
        *,
        query_vector: np.ndarray,
        mode: str,
        model_name: str,
        top_k: int,
        filters: dict[str, Any] | None,
        similarity_threshold: float = 0.9,
    ) -> dict[str, Any] | None:
        filters_key = json.dumps(filters or {}, sort_keys=True)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT query_vector, response_json
                FROM semantic_query_cache
                WHERE mode = ? AND model_name = ? AND top_k = ? AND filters_json = ?
                """,
                (mode, model_name, top_k, filters_key),
            ).fetchall()
        if not rows:
            return None

        best_score = -1.0
        best_response: dict[str, Any] | None = None
        query = query_vector.astype(np.float32).reshape(-1)
        for vec_json, response_json in rows:
            cached_vec = np.asarray(json.loads(vec_json), dtype=np.float32).reshape(-1)
            score = self._cosine_similarity(query, cached_vec)
            if score > best_score:
                best_score = score
                best_response = json.loads(response_json)
        if best_score >= similarity_threshold:
            return best_response
        return None

    def set(
        self,
        *,
        query_vector: np.ndarray,
        mode: str,
        model_name: str,
        top_k: int,
        filters: dict[str, Any] | None,
        response_payload: dict[str, Any],
    ) -> None:
        filters_key = json.dumps(filters or {}, sort_keys=True)
        signature = hash_text(
            json.dumps(
                {
                    "mode": mode,
                    "model_name": model_name,
                    "top_k": top_k,
                    "filters": filters or {},
                    "query_vector_head": query_vector[:16].tolist(),
                },
                sort_keys=True,
            )
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO semantic_query_cache(
                    cache_id, mode, model_name, top_k, filters_json, query_vector, response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signature,
                    mode,
                    model_name,
                    top_k,
                    filters_key,
                    json.dumps(query_vector.astype(float).tolist()),
                    json.dumps(response_payload),
                ),
            )
