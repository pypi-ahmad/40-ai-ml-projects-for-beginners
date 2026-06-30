from __future__ import annotations


def test_schema_report_contains_core_tables(schema_report: dict) -> None:
    tables = schema_report["tables"]
    for required in ["customers", "orders", "order_details", "products"]:
        assert required in tables


def test_schema_report_relationships_not_empty(schema_report: dict) -> None:
    relationships = schema_report["relationships"]
    assert len(relationships) > 0
