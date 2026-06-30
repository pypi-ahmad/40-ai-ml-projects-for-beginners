from __future__ import annotations

from ai_sql_assistant.validation.validator import SQLValidator


def test_validator_accepts_valid_select(schema_report: dict) -> None:
    validator = SQLValidator(schema_report)
    sql = "SELECT customer_id, COUNT(*) AS c FROM orders GROUP BY customer_id LIMIT 10"
    report = validator.validate(sql)

    assert report.is_valid is True
    assert not any(issue.code == "blocked_keyword" for issue in report.issues)


def test_validator_blocks_dml(schema_report: dict) -> None:
    validator = SQLValidator(schema_report)
    report = validator.validate("DELETE FROM orders")

    assert report.is_valid is False
    assert any(issue.code == "blocked_keyword" for issue in report.issues)


def test_validator_flags_unknown_table(schema_report: dict) -> None:
    validator = SQLValidator(schema_report)
    report = validator.validate("SELECT * FROM imaginary_table")

    assert report.is_valid is False
    assert any(issue.code == "unknown_table" for issue in report.issues)


def test_validator_allows_select_alias_in_group_order(schema_report: dict) -> None:
    validator = SQLValidator(schema_report)
    sql = """
    SELECT STRFTIME('%Y-%m', order_date) AS month, COUNT(*) AS total_orders
    FROM orders
    GROUP BY month
    ORDER BY month
    """
    report = validator.validate(sql)
    assert report.is_valid is True


def test_validator_blocks_empty_or_invalid_sql(schema_report: dict) -> None:
    validator = SQLValidator(schema_report)
    report = validator.validate("")
    assert report.is_valid is False
