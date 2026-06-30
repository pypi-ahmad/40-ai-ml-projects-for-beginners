"""Rule-based follow-up resolver for context-aware SQL refinement."""

from __future__ import annotations

import re


def resolve_followup_sql(question: str, previous_sql: str | None) -> str | None:
    """Rewrite previous SQL for common follow-up patterns.

    Supported pattern:
    - "Only for <value>"
    - "Only for <value> region/market/country"
    """
    if not previous_sql:
        return None

    q = question.strip().lower()
    match = re.match(r"^(only\s+for|for\s+only)\s+(.+?)[\.?]?$", q)
    if not match:
        return None

    raw_value = match.group(2).strip()
    value = raw_value.title()

    target_column = _detect_best_filter_column(previous_sql)
    if not target_column:
        return None

    condition = f"{target_column} = '{value}'"
    return _inject_condition(previous_sql, condition)


def _detect_best_filter_column(sql: str) -> str | None:
    low = sql.lower()

    if "o.market" in low:
        return "o.market"
    if re.search(r"\bmarket\b", low):
        return "market"

    if "c.country" in low:
        return "c.country"
    if "o.ship_country" in low:
        return "o.ship_country"
    if re.search(r"\bcountry\b", low):
        return "country"

    return None


def _inject_condition(sql: str, condition: str) -> str:
    lower = sql.lower()

    clause_positions = [
        idx
        for keyword in [" group by ", " order by ", " limit "]
        if (idx := lower.find(keyword)) != -1
    ]
    clause_start = min(clause_positions) if clause_positions else len(sql)

    where_pos = lower.find(" where ")
    if where_pos != -1 and where_pos < clause_start:
        return sql[:clause_start] + f" AND {condition}" + sql[clause_start:]

    return sql[:clause_start] + f" WHERE {condition}" + sql[clause_start:]
