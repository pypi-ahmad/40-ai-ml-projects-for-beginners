"""CSV reader alias."""

from __future__ import annotations

from crew_platform.tools.csv_analyzer import CSVAnalyzerTool


class CSVReaderTool(CSVAnalyzerTool):
    name = "csv_reader"
    description = "Read CSV schema and summary stats"
