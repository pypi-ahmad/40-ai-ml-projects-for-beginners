#!/usr/bin/env python3
"""Generate 100 benchmark cases for SQL assistant evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from ai_sql_assistant.logging_utils import configure_logging, logger

def build_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    case_id = 1

    markets = ["Europe", "North America", "APAC", "LATAM"]
    countries = ["USA", "Germany", "UK", "France", "India", "Canada", "Brazil", "Japan", "Australia"]
    segments = ["Enterprise", "SMB", "Mid-Market", "Public Sector"]

    # 1) Monthly revenue trend by market.
    for market in markets:
        for year in [2020, 2021, 2022, 2023, 2024]:
            sql = f"""
            SELECT
                strftime('%Y-%m', o.order_date) AS month,
                ROUND(SUM(od.quantity * od.unit_price * (1 - od.discount)), 2) AS revenue
            FROM orders o
            JOIN order_details od ON o.order_id = od.order_id
            WHERE o.market = '{market}'
              AND strftime('%Y', o.order_date) = '{year}'
            GROUP BY strftime('%Y-%m', o.order_date)
            ORDER BY month;
            """.strip()
            cases.append(
                {
                    "case_id": f"B{case_id:03d}",
                    "category": "trend_analysis",
                    "question": f"Show monthly net revenue for {market} in {year}.",
                    "ground_truth_sql": sql,
                    "tags": ["trend", "revenue", "group_by", "date"],
                }
            )
            case_id += 1

    # 2) Top customers by revenue per country.
    for country in countries:
        sql = f"""
        SELECT
            c.customer_id,
            c.company_name,
            ROUND(SUM(od.quantity * od.unit_price * (1 - od.discount)), 2) AS revenue
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        JOIN order_details od ON o.order_id = od.order_id
        WHERE c.country = '{country}'
        GROUP BY c.customer_id, c.company_name
        ORDER BY revenue DESC
        LIMIT 10;
        """.strip()
        cases.append(
            {
                "case_id": f"B{case_id:03d}",
                "category": "top_n",
                "question": f"Who are top 10 customers by net revenue in {country}?",
                "ground_truth_sql": sql,
                "tags": ["top_n", "customer", "revenue"],
            }
        )
        case_id += 1

    # 3) Product category performance by segment.
    for seg in segments:
        for year in [2021, 2022, 2023, 2024]:
            sql = f"""
            SELECT
                cat.category_name,
                ROUND(SUM(od.quantity * od.unit_price * (1 - od.discount)), 2) AS revenue,
                SUM(od.quantity) AS units
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            JOIN order_details od ON o.order_id = od.order_id
            JOIN products p ON od.product_id = p.product_id
            JOIN categories cat ON p.category_id = cat.category_id
            WHERE c.segment = '{seg}'
              AND strftime('%Y', o.order_date) = '{year}'
            GROUP BY cat.category_name
            ORDER BY revenue DESC;
            """.strip()
            cases.append(
                {
                    "case_id": f"B{case_id:03d}",
                    "category": "segment_analysis",
                    "question": f"Break down {year} net revenue by product category for {seg} customers.",
                    "ground_truth_sql": sql,
                    "tags": ["join", "segment", "category", "group_by"],
                }
            )
            case_id += 1

    # 4) Window/ranking queries per market.
    for market in markets:
        for year in [2022, 2023, 2024]:
            sql = f"""
            WITH customer_rev AS (
                SELECT
                    c.customer_id,
                    c.company_name,
                    ROUND(SUM(od.quantity * od.unit_price * (1 - od.discount)), 2) AS revenue
                FROM customers c
                JOIN orders o ON c.customer_id = o.customer_id
                JOIN order_details od ON o.order_id = od.order_id
                WHERE o.market = '{market}'
                  AND strftime('%Y', o.order_date) = '{year}'
                GROUP BY c.customer_id, c.company_name
            )
            SELECT
                customer_id,
                company_name,
                revenue,
                DENSE_RANK() OVER (ORDER BY revenue DESC) AS revenue_rank
            FROM customer_rev
            ORDER BY revenue_rank, customer_id
            LIMIT 20;
            """.strip()
            cases.append(
                {
                    "case_id": f"B{case_id:03d}",
                    "category": "window_functions",
                    "question": f"Rank top customers in {market} for {year} by net revenue.",
                    "ground_truth_sql": sql,
                    "tags": ["cte", "window", "ranking"],
                }
            )
            case_id += 1

    # 5) Inventory and supplier queries.
    for stock_limit in [20, 40, 60, 80, 100]:
        for category in ["Beverages", "Confections", "Seafood", "Produce"]:
            sql = f"""
            SELECT
                p.product_id,
                p.product_name,
                p.units_in_stock,
                s.company_name AS supplier,
                cat.category_name
            FROM products p
            JOIN suppliers s ON p.supplier_id = s.supplier_id
            JOIN categories cat ON p.category_id = cat.category_id
            WHERE cat.category_name = '{category}'
              AND p.units_in_stock <= {stock_limit}
              AND p.discontinued = 0
            ORDER BY p.units_in_stock ASC, p.product_id
            LIMIT 25;
            """.strip()
            cases.append(
                {
                    "case_id": f"B{case_id:03d}",
                    "category": "inventory",
                    "question": f"List low-stock active {category} products at or below {stock_limit} units with suppliers.",
                    "ground_truth_sql": sql,
                    "tags": ["inventory", "join", "filter"],
                }
            )
            case_id += 1

    # 6) Employee performance.
    for year in [2021, 2022, 2023, 2024]:
        for market in markets:
            sql = f"""
            SELECT
                e.employee_id,
                e.first_name || ' ' || e.last_name AS employee_name,
                COUNT(DISTINCT o.order_id) AS orders_handled,
                ROUND(SUM(od.quantity * od.unit_price * (1 - od.discount)), 2) AS revenue
            FROM employees e
            JOIN orders o ON e.employee_id = o.employee_id
            JOIN order_details od ON o.order_id = od.order_id
            WHERE strftime('%Y', o.order_date) = '{year}'
              AND o.market = '{market}'
            GROUP BY e.employee_id, employee_name
            HAVING COUNT(DISTINCT o.order_id) >= 20
            ORDER BY revenue DESC
            LIMIT 15;
            """.strip()
            cases.append(
                {
                    "case_id": f"B{case_id:03d}",
                    "category": "employee_performance",
                    "question": f"Show high-performing employees in {market} during {year}, with at least 20 orders.",
                    "ground_truth_sql": sql,
                    "tags": ["having", "employee", "kpi"],
                }
            )
            case_id += 1

    # 7) Comparative queries.
    for country in countries[:8]:
        sql = f"""
        WITH monthly AS (
            SELECT
                strftime('%Y-%m', o.order_date) AS month,
                ROUND(SUM(od.quantity * od.unit_price * (1 - od.discount)), 2) AS revenue
            FROM orders o
            JOIN order_details od ON o.order_id = od.order_id
            WHERE o.ship_country = '{country}'
            GROUP BY strftime('%Y-%m', o.order_date)
        )
        SELECT
            month,
            revenue,
            revenue - LAG(revenue) OVER (ORDER BY month) AS mom_change
        FROM monthly
        ORDER BY month;
        """.strip()
        cases.append(
            {
                "case_id": f"B{case_id:03d}",
                "category": "comparison",
                "question": f"For {country}, show month-over-month revenue change.",
                "ground_truth_sql": sql,
                "tags": ["comparison", "lag", "window"],
            }
        )
        case_id += 1

    # 8) CTE + subquery KPI checks.
    for seg in segments:
        for year in [2022, 2023, 2024]:
            sql = f"""
            WITH customer_rev AS (
                SELECT
                    c.customer_id,
                    ROUND(SUM(od.quantity * od.unit_price * (1 - od.discount)), 2) AS revenue
                FROM customers c
                JOIN orders o ON c.customer_id = o.customer_id
                JOIN order_details od ON o.order_id = od.order_id
                WHERE c.segment = '{seg}'
                  AND strftime('%Y', o.order_date) = '{year}'
                GROUP BY c.customer_id
            )
            SELECT
                AVG(revenue) AS avg_revenue,
                MAX(revenue) AS max_revenue,
                MIN(revenue) AS min_revenue,
                SUM(CASE WHEN revenue > (SELECT AVG(revenue) FROM customer_rev) THEN 1 ELSE 0 END) AS above_avg_customers
            FROM customer_rev;
            """.strip()
            cases.append(
                {
                    "case_id": f"B{case_id:03d}",
                    "category": "kpi",
                    "question": f"For {seg} customers in {year}, summarize average/min/max customer revenue and count above average.",
                    "ground_truth_sql": sql,
                    "tags": ["cte", "subquery", "kpi"],
                }
            )
            case_id += 1

    # ensure exactly 100
    return cases[:100]


def main() -> None:
    configure_logging()
    out_path = Path("benchmarks/benchmark_cases.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    if len(cases) < 100:
        raise RuntimeError(f"Expected at least 100 benchmark cases, got {len(cases)}")
    out_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    logger.info("Wrote {} cases to {}", len(cases), out_path)


if __name__ == "__main__":
    main()
