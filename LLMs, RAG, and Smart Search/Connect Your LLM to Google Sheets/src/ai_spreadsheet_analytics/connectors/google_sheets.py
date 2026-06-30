"""Google Sheets ingestion with caching and incremental refresh."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gspread
import pandas as pd
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ai_spreadsheet_analytics.schemas import DatasetBundle, DatasetFrame, SheetMetadata
from ai_spreadsheet_analytics.state_store import SQLiteStateStore
from ai_spreadsheet_analytics.utils import stable_df_hash


@dataclass(slots=True)
class SheetLoadRequest:
    """Load request for one worksheet."""

    spreadsheet_id: str
    worksheet_title: str


class GoogleSheetsLoader:
    """Google Sheets loader with caching and metadata inspection."""

    def __init__(self, client: gspread.Client, cache_dir: Path, state_store: SQLiteStateStore) -> None:
        self.client = client
        self.cache_dir = cache_dir
        self.state_store = state_store
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def inspect_spreadsheet(self, spreadsheet_id: str) -> list[dict[str, Any]]:
        """Inspect sheet and return worksheet metadata overview."""
        workbook = self.client.open_by_key(spreadsheet_id)
        details: list[dict[str, Any]] = []
        for worksheet in workbook.worksheets():
            details.append(
                {
                    "spreadsheet_id": spreadsheet_id,
                    "spreadsheet_title": workbook.title,
                    "worksheet_title": worksheet.title,
                    "rows": worksheet.row_count,
                    "columns": worksheet.col_count,
                }
            )
        return details

    def _cache_key(self, spreadsheet_id: str, worksheet_title: str) -> str:
        return f"{spreadsheet_id}__{worksheet_title}".replace("/", "_")

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.parquet"

    def _metadata_from_df(
        self,
        spreadsheet_id: str,
        spreadsheet_title: str,
        worksheet_title: str,
        dataframe: pd.DataFrame,
    ) -> SheetMetadata:
        return SheetMetadata(
            spreadsheet_id=spreadsheet_id,
            spreadsheet_title=spreadsheet_title,
            worksheet_title=worksheet_title,
            rows=len(dataframe),
            columns=len(dataframe.columns),
            dtypes={col: str(dtype) for col, dtype in dataframe.dtypes.items()},
            missing_by_column=dataframe.isna().sum().astype(int).to_dict(),
            sample_records=dataframe.head(5).to_dict(orient="records"),
        )

    def _fetch_df(self, spreadsheet_id: str, worksheet_title: str) -> tuple[pd.DataFrame, str]:
        workbook = self.client.open_by_key(spreadsheet_id)
        worksheet = workbook.worksheet(worksheet_title)
        records = worksheet.get_all_records()
        dataframe = pd.DataFrame(records)
        return dataframe, workbook.title

    def load_worksheet(
        self,
        spreadsheet_id: str,
        worksheet_title: str,
        use_cache: bool = True,
    ) -> DatasetFrame:
        """Load one worksheet with caching and incremental metadata tracking."""
        cache_key = self._cache_key(spreadsheet_id, worksheet_title)
        cache_path = self._cache_path(cache_key)

        if use_cache and cache_path.exists():
            try:
                cached_df = pd.read_parquet(cache_path)
                logger.info("Loaded worksheet from cache: {}", cache_key)
                metadata = self._metadata_from_df(spreadsheet_id, "cached", worksheet_title, cached_df)
                return DatasetFrame(key=cache_key, dataframe=cached_df, metadata=metadata)
            except Exception:  # noqa: BLE001
                logger.warning("Cache load failed; refetching from API for {}", cache_key)

        df, spreadsheet_title = self._fetch_df(spreadsheet_id, worksheet_title)
        row_hash = stable_df_hash(df)

        if use_cache:
            df.to_parquet(cache_path, index=False)
            self.state_store.upsert_cache_manifest(cache_key, spreadsheet_id, worksheet_title, row_hash)

        metadata = self._metadata_from_df(spreadsheet_id, spreadsheet_title, worksheet_title, df)
        return DatasetFrame(key=cache_key, dataframe=df, metadata=metadata)

    def load_batch(self, requests: list[SheetLoadRequest], use_cache: bool = True) -> DatasetBundle:
        """Load multiple worksheets across spreadsheets."""
        bundle = DatasetBundle()
        for request in requests:
            frame = self.load_worksheet(
                spreadsheet_id=request.spreadsheet_id,
                worksheet_title=request.worksheet_title,
                use_cache=use_cache,
            )
            bundle.frames[frame.key] = frame
        return bundle

    @staticmethod
    def incremental_diff(
        old_df: pd.DataFrame,
        new_df: pd.DataFrame,
        key_columns: list[str] | None = None,
    ) -> dict[str, int]:
        """Compute incremental refresh summary.

        Args:
            old_df: Previous snapshot.
            new_df: Latest snapshot.
            key_columns: Optional row identifiers.

        Returns:
            Counts of inserted/updated/deleted/unchanged rows.
        """
        if old_df.empty:
            return {
                "inserted": len(new_df),
                "updated": 0,
                "deleted": 0,
                "unchanged": 0,
            }

        if key_columns and set(key_columns).issubset(old_df.columns) and set(key_columns).issubset(new_df.columns):
            old_idx = old_df.set_index(key_columns)
            new_idx = new_df.set_index(key_columns)
        else:
            old_idx = old_df.reset_index(drop=True)
            new_idx = new_df.reset_index(drop=True)

        old_hashes = old_idx.fillna("<NA>").astype(str).agg("|".join, axis=1)
        new_hashes = new_idx.fillna("<NA>").astype(str).agg("|".join, axis=1)

        old_set = set(old_hashes)
        new_set = set(new_hashes)
        inserted = len(new_set - old_set)
        deleted = len(old_set - new_set)
        unchanged = len(new_set & old_set)
        updated = max(len(new_hashes) - unchanged - inserted, 0)
        return {
            "inserted": inserted,
            "updated": updated,
            "deleted": deleted,
            "unchanged": unchanged,
        }
