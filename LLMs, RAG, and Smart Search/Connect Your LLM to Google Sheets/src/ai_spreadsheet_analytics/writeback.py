"""Google Sheets write-back service."""

from __future__ import annotations

import pandas as pd

from ai_spreadsheet_analytics.connectors.google_sheets import GoogleSheetsLoader


class GoogleSheetsWriteBack:
    """Write outputs to new worksheets with safe defaults."""

    def __init__(self, loader: GoogleSheetsLoader) -> None:
        self.loader = loader

    def write_dataframe(
        self,
        spreadsheet_id: str,
        worksheet_title: str,
        df: pd.DataFrame,
        overwrite: bool = False,
    ) -> dict[str, int | str]:
        """Write dataframe to worksheet.

        Args:
            spreadsheet_id: Destination spreadsheet ID.
            worksheet_title: Target worksheet title.
            df: Dataframe to write.
            overwrite: If False, never overwrite existing worksheet.

        Returns:
            Write metadata.
        """
        workbook = self.loader.client.open_by_key(spreadsheet_id)
        existing = None
        for worksheet in workbook.worksheets():
            if worksheet.title == worksheet_title:
                existing = worksheet
                break

        if existing and not overwrite:
            worksheet_title = f"{worksheet_title}_new"
            existing = None

        if existing is None:
            existing = workbook.add_worksheet(
                title=worksheet_title,
                rows=max(len(df) + 20, 200),
                cols=max(len(df.columns) + 5, 20),
            )
        else:
            existing.clear()

        values = [df.columns.tolist()] + df.fillna("").astype(str).values.tolist()
        existing.update(values)
        return {
            "spreadsheet_id": spreadsheet_id,
            "worksheet_title": worksheet_title,
            "rows_written": len(df),
            "columns_written": len(df.columns),
        }

    def write_insights(
        self,
        spreadsheet_id: str,
        worksheet_title: str,
        insights: list[str],
        overwrite: bool = False,
    ) -> dict[str, int | str]:
        """Write insight list to sheet."""
        df = pd.DataFrame({"insight": insights})
        return self.write_dataframe(spreadsheet_id, worksheet_title, df, overwrite=overwrite)
