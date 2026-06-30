"""Execute notebook sequentially without manual steps."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_notebook", default="notebooks/zero_to_hero_api_intelligence_agent.ipynb")
    parser.add_argument(
        "--output",
        default="notebooks/zero_to_hero_api_intelligence_agent.executed.ipynb",
    )
    args = parser.parse_args()
    output_path = args.output
    output_dir = ""
    output_name = output_path
    if "/" in output_path:
        output_dir, output_name = output_path.rsplit("/", 1)

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        args.input_notebook,
        "--output",
        output_name,
    ]
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
