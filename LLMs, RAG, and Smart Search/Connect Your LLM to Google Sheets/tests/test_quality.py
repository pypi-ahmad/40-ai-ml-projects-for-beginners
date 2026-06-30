from __future__ import annotations

import pandas as pd

from ai_spreadsheet_analytics.quality import DataQualityProfiler


def test_quality_profiler_detects_missing_and_duplicates() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "bad-date", None],
            "revenue": [100, 100, "oops", 130],
            "region": ["US", "US", "UK", None],
        }
    )
    report = DataQualityProfiler().profile("test", df)

    issue_names = {issue.check_name for issue in report.issues}
    assert "missing_values" in issue_names
    assert "duplicate_rows" in issue_names
    assert report.metrics["row_count"] == 4
