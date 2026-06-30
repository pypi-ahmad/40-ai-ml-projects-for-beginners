"""Business glossary mapping for schema-aware query generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GlossaryTerm:
    """Glossary mapping from business term to schema hints."""

    term: str
    tables: list[str]
    columns: list[str]
    note: str


DEFAULT_GLOSSARY: dict[str, GlossaryTerm] = {
    "customers": GlossaryTerm(
        term="customers",
        tables=["customers", "orders"],
        columns=["customer_id", "company_name", "segment", "country"],
        note="Customer dimension with geography and segment context.",
    ),
    "sales": GlossaryTerm(
        term="sales",
        tables=["orders", "order_details", "products"],
        columns=["order_date", "quantity", "unit_price", "discount", "freight"],
        note="Revenue proxy = sum(quantity * unit_price * (1 - discount)).",
    ),
    "revenue": GlossaryTerm(
        term="revenue",
        tables=["orders", "order_details"],
        columns=["quantity", "unit_price", "discount", "order_date"],
        note="Use net sales formula with discount adjustment.",
    ),
    "profit": GlossaryTerm(
        term="profit",
        tables=["order_details", "products"],
        columns=["unit_price", "quantity", "units_in_stock"],
        note="No true COGS field; use proxy and mention limitation.",
    ),
    "inventory": GlossaryTerm(
        term="inventory",
        tables=["products", "suppliers", "categories"],
        columns=["units_in_stock", "discontinued", "category_id"],
        note="Inventory stock sits in products table.",
    ),
    "hr": GlossaryTerm(
        term="hr",
        tables=["employees", "orders"],
        columns=["employee_id", "hire_date", "country"],
        note="Employee performance uses orders linked to employee_id.",
    ),
    "marketing": GlossaryTerm(
        term="marketing",
        tables=["customers", "orders"],
        columns=["segment", "country", "order_date"],
        note="Segment and geo rollups represent campaign targeting proxies.",
    ),
}


def glossary_context(question: str, glossary: dict[str, GlossaryTerm] | None = None) -> str:
    """Select glossary hints relevant to question."""
    glossary = glossary or DEFAULT_GLOSSARY
    q = question.lower()
    hits = [item for key, item in glossary.items() if key in q]
    if not hits:
        return "No glossary terms matched."

    lines: list[str] = []
    for item in hits:
        lines.append(
            f"- {item.term}: tables={', '.join(item.tables)}; "
            f"columns={', '.join(item.columns)}; note={item.note}"
        )
    return "\n".join(lines)
