"""Generate benchmark question set (100 cases)."""

from __future__ import annotations

import json
from pathlib import Path

DOMAINS = ["sales", "retail", "hr", "finance", "marketing", "healthcare", "support", "ecommerce"]
PATTERNS = [
    ("Which product generated highest revenue?", "categorical", ["product", "revenue"]),
    ("Which month had highest revenue?", "time_series", ["month", "revenue"]),
    ("Show declining trend periods.", "time_series", ["decline", "trend"]),
    ("What changed last quarter?", "comparative", ["quarter", "change"]),
    ("Explain anomalies in revenue.", "anomaly", ["anomaly", "revenue"]),
    ("Top 5 customers by sales.", "categorical", ["top", "customer"]),
    ("Bottom 5 products by sales.", "categorical", ["bottom", "product"]),
    ("Summarize key business risks.", "summary", ["risk", "recommendation"]),
    ("What is conversion trend?", "time_series", ["conversion", "trend"]),
    ("Which channel has best ROI?", "comparative", ["channel", "roi"]),
]


def main() -> None:
    cases = []
    idx = 1
    while len(cases) < 100:
        for domain in DOMAINS:
            for q, expected_type, keywords in PATTERNS:
                if len(cases) >= 100:
                    break
                case = {
                    "case_id": f"C{idx:03d}",
                    "question": f"[{domain}] {q}",
                    "expected_type": expected_type,
                    "expected_keywords": keywords,
                    "domain": domain,
                }
                cases.append(case)
                idx += 1
            if len(cases) >= 100:
                break

    root = Path(__file__).resolve().parents[1]
    output = root / "data" / "benchmarks" / "questions.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"Generated {len(cases)} cases -> {output}")


if __name__ == "__main__":
    main()
