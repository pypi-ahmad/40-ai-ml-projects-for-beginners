"""SQL safety and semantic validation engine."""

from __future__ import annotations

import re
from typing import Any

import sqlglot
from sqlglot import exp

from ai_sql_assistant.constants import BLOCKED_SQL_KEYWORDS
from ai_sql_assistant.types import ValidationIssue, ValidationReport
from ai_sql_assistant.utils.sql_utils import normalize_sql


class SQLValidator:
    """Validate generated SQL before execution."""

    def __init__(
        self,
        schema_report: dict[str, Any],
        allow_multi_statement: bool = False,
        strict_join_checks: bool = True,
    ) -> None:
        self.schema_report = schema_report
        self.allow_multi_statement = allow_multi_statement
        self.strict_join_checks = strict_join_checks

        self.tables = set(schema_report.get("tables", {}).keys())
        self.table_columns = {
            table: {col["name"].lower() for col in meta["columns"]}
            for table, meta in schema_report.get("tables", {}).items()
        }
        self.all_columns = set().union(*self.table_columns.values()) if self.table_columns else set()
        self.relationship_pairs = {
            tuple(sorted((rel["from_table"], rel["to_table"])))
            for rel in schema_report.get("relationships", [])
        }

    def validate(self, sql: str) -> ValidationReport:
        """Run all SQL validations and return report."""
        normalized = normalize_sql(sql)
        issues: list[ValidationIssue] = []

        stripped = sql.strip()
        lowered = stripped.lower()

        # Prompt/SQL injection markers.
        if "--" in lowered or "/*" in lowered or "*/" in lowered:
            issues.append(
                ValidationIssue(
                    code="suspicious_comment",
                    message="Comments are blocked to reduce prompt/SQL injection risk.",
                )
            )

        blocked_hits = sorted({kw for kw in BLOCKED_SQL_KEYWORDS if re.search(rf"\b{kw}\b", lowered)})
        if blocked_hits:
            issues.append(
                ValidationIssue(
                    code="blocked_keyword",
                    message=f"Blocked SQL keywords found: {', '.join(blocked_hits)}",
                )
            )

        statements: list[exp.Expression]
        try:
            statements = sqlglot.parse(stripped, read="sqlite")
        except Exception as exc:
            issues.append(ValidationIssue(code="syntax_error", message=f"SQL parse error: {exc}"))
            return ValidationReport(
                is_valid=False,
                normalized_sql=normalized,
                issues=issues,
                blocked_keywords=blocked_hits,
            )

        if not self.allow_multi_statement and len(statements) != 1:
            issues.append(
                ValidationIssue(
                    code="multi_statement_blocked",
                    message="Only single statement SQL is allowed.",
                )
            )

        if not statements:
            issues.append(ValidationIssue(code="empty_sql", message="Empty SQL provided."))
            return ValidationReport(
                is_valid=False,
                normalized_sql=normalized,
                issues=issues,
                blocked_keywords=blocked_hits,
            )

        root = statements[0]
        if root is None or not isinstance(root, exp.Expression):
            issues.append(
                ValidationIssue(
                    code="invalid_statement",
                    message="Unable to parse SQL statement into valid AST.",
                )
            )
            return ValidationReport(
                is_valid=False,
                normalized_sql=normalized,
                issues=issues,
                blocked_keywords=blocked_hits,
            )

        if not root.find(exp.Select):
            issues.append(
                ValidationIssue(
                    code="non_select_statement",
                    message="Only SELECT queries are allowed.",
                )
            )

        self._validate_identifiers(root, issues)
        self._validate_joins(root, issues)

        return ValidationReport(
            is_valid=not any(issue.severity == "error" for issue in issues),
            normalized_sql=normalized,
            issues=issues,
            blocked_keywords=blocked_hits,
        )

    def _validate_identifiers(self, root: exp.Expression, issues: list[ValidationIssue]) -> None:
        tables_seen: set[str] = set()
        select_aliases: set[str] = set()

        for select in root.find_all(exp.Select):
            for item in select.expressions:
                alias = item.alias_or_name
                if alias:
                    select_aliases.add(alias.lower())

        for table in root.find_all(exp.Table):
            name = table.name
            if not name:
                continue
            if name not in self.tables:
                issues.append(
                    ValidationIssue(
                        code="unknown_table",
                        message=f"Unknown table: {name}",
                    )
                )
            else:
                tables_seen.add(name)

        for column in root.find_all(exp.Column):
            col_name = (column.name or "").lower()
            table_name = column.table
            if not col_name:
                continue

            if table_name:
                if table_name in self.table_columns and col_name not in self.table_columns[table_name]:
                    issues.append(
                        ValidationIssue(
                            code="unknown_column",
                            message=f"Unknown column `{table_name}.{col_name}`",
                        )
                    )
            else:
                if col_name in select_aliases:
                    continue
                if col_name not in self.all_columns and col_name != "*":
                    issues.append(
                        ValidationIssue(
                            code="unknown_column",
                            message=f"Unknown unqualified column `{col_name}`",
                        )
                    )

    def _validate_joins(self, root: exp.Expression, issues: list[ValidationIssue]) -> None:
        tables = [table.name for table in root.find_all(exp.Table) if table.name]
        joins = list(root.find_all(exp.Join))

        if len(set(tables)) > 1 and not joins:
            issues.append(
                ValidationIssue(
                    code="missing_join",
                    message="Multiple tables used without explicit JOIN.",
                )
            )

        for join in joins:
            if join.args.get("on") is None:
                issues.append(
                    ValidationIssue(
                        code="join_without_on",
                        message="JOIN detected without ON condition.",
                    )
                )

        if not self.strict_join_checks:
            return

        # Heuristic: joined table pairs should have known relationship.
        chain: list[str] = []
        from_clause = root.args.get("from")
        if from_clause and from_clause.this and isinstance(from_clause.this, exp.Table):
            chain.append(from_clause.this.name)

        for join in joins:
            joined = join.this
            if isinstance(joined, exp.Table):
                chain.append(joined.name)

        for idx in range(1, len(chain)):
            pair = tuple(sorted((chain[idx - 1], chain[idx])))
            if pair not in self.relationship_pairs:
                issues.append(
                    ValidationIssue(
                        code="ambiguous_join",
                        message=(
                            f"Join pair `{chain[idx - 1]}` <-> `{chain[idx]}` not found in FK graph. "
                            "Verify join path."
                        ),
                        severity="warning",
                    )
                )
