from __future__ import annotations

from ai_sql_assistant.utils.sql_utils import clean_sql_text, fix_common_sqlite_patterns


def test_clean_sql_text_removes_labels_and_fences() -> None:
    text = """
    SQLQuery:
    ```sql
    SELECT * FROM orders LIMIT 5;
    ```
    """
    cleaned = clean_sql_text(text)
    assert cleaned.lower().startswith("select")
    assert "sqlquery" not in cleaned.lower()


def test_fix_strftime_year_literals() -> None:
    sql = "SELECT * FROM orders WHERE STRFTIME('%Y', order_date) = 2024"
    fixed = fix_common_sqlite_patterns(sql)
    assert " = '2024'" in fixed


def test_fix_month_year_functions_for_sqlite() -> None:
    sql = "SELECT MONTH(order_date) AS m, YEAR(order_date) AS y FROM orders"
    fixed = fix_common_sqlite_patterns(sql)
    assert "STRFTIME('%m'" in fixed
    assert "STRFTIME('%Y'" in fixed
