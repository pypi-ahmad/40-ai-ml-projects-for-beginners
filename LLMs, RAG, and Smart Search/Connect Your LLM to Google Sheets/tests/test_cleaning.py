from __future__ import annotations

import pandas as pd

from ai_spreadsheet_analytics.cleaning import DataCleaner
from ai_spreadsheet_analytics.schemas import CleaningStrategy


def test_cleaning_parses_currency_and_percentages() -> None:
    df = pd.DataFrame(
        {
            "amount": ["$1,000", "$2,500", None],
            "rate": ["10%", "25%", None],
            "category": ["A", "A", "A"],
        }
    )

    strategy = CleaningStrategy(missing_value_strategy="zero", drop_constant_columns=True)
    result = DataCleaner().clean("sample", df, strategy)

    assert "amount" in result.cleaned.columns
    assert result.cleaned["amount"].iloc[0] == 1000.0
    assert result.cleaned["rate"].iloc[0] == 0.10
    assert "category" not in result.cleaned.columns
