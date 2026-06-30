from __future__ import annotations

from ai_sql_assistant.memory.followup import resolve_followup_sql


def test_followup_rewrite_injects_market_filter() -> None:
    previous = (
        "SELECT STRFTIME('%Y-%m', o.order_date) AS month, o.market, "
        "SUM(od.unit_price * od.quantity) AS revenue "
        "FROM orders o JOIN order_details od ON o.order_id = od.order_id "
        "GROUP BY month, o.market ORDER BY month"
    )
    rewritten = resolve_followup_sql("Only for Europe.", previous)
    assert rewritten is not None
    assert "o.market = 'Europe'" in rewritten


def test_followup_rewrite_none_for_non_followup() -> None:
    previous = "SELECT * FROM orders"
    rewritten = resolve_followup_sql("Show top customers", previous)
    assert rewritten is None
