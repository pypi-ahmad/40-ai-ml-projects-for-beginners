"""Run local verification suite for multimodal platform."""

from __future__ import annotations

import subprocess
from pathlib import Path

CHECKS = [
    ["uv", "run", "ruff", "check", "src", "tests", "scripts", "app.py", "mcp_server.py"],
    ["uv", "run", "mypy", "src"],
    ["uv", "run", "pytest", "tests/unit", "tests/integration"],
]


def run_check(cmd: list[str]) -> tuple[int, str]:
    process = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = (process.stdout or "") + (process.stderr or "")
    return process.returncode, output


def main() -> None:
    report_lines = ["# Verification Report", ""]

    for cmd in CHECKS:
        code, output = run_check(cmd)
        report_lines.append(f"## {' '.join(cmd)}")
        report_lines.append(f"- exit_code: {code}")
        report_lines.append("```text")
        report_lines.append(output.strip()[:4000])
        report_lines.append("```")
        report_lines.append("")

    report_path = Path("outputs/reports/verification_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Verification report saved: {report_path}")


if __name__ == "__main__":
    main()
