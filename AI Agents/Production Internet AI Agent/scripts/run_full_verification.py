"""Tiered verification script for smoke and full checks."""

from __future__ import annotations

import argparse
import subprocess

SMOKE_COMMANDS = [
    ["uv", "run", "pytest", "tests/test_config.py", "tests/test_tools.py", "-q"],
    ["uv", "run", "pytest", "tests/test_api.py", "-q"],
]

FULL_COMMANDS = [
    ["uv", "run", "pytest", "-q"],
]


def run_commands(commands: list[list[str]]) -> None:
    for command in commands:
        print("$", " ".join(command))
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    if args.mode == "smoke":
        run_commands(SMOKE_COMMANDS)
    else:
        run_commands(SMOKE_COMMANDS + FULL_COMMANDS)


if __name__ == "__main__":
    main()
