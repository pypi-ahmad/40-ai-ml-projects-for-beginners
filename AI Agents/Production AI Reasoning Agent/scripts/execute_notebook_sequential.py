"""Execute a notebook sequentially without Jupyter kernel sockets.

This is used for restricted environments where kernel socket binding is blocked.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import io
from pathlib import Path

import nbformat
from nbformat import v4 as nbf_v4


def execute_notebook(input_path: Path, output_path: Path) -> None:
    nb = nbformat.read(input_path, as_version=4)
    global_ns: dict[str, object] = {"__name__": "__notebook__"}
    execution_count = 1

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                code_obj = compile(
                    cell.source,
                    filename=str(input_path),
                    mode="exec",
                    flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                )
                result = eval(code_obj, global_ns)  # noqa: S307
                if asyncio.iscoroutine(result):
                    asyncio.run(result)

            outputs = []
            std_out = stdout_buffer.getvalue()
            std_err = stderr_buffer.getvalue()
            if std_out:
                outputs.append(nbf_v4.new_output(output_type="stream", name="stdout", text=std_out))
            if std_err:
                outputs.append(nbf_v4.new_output(output_type="stream", name="stderr", text=std_err))
            cell.outputs = outputs
            cell.execution_count = execution_count
            execution_count += 1

        except Exception as exc:  # noqa: BLE001
            cell.outputs = [
                nbf_v4.new_output(
                    output_type="error",
                    ename=exc.__class__.__name__,
                    evalue=str(exc),
                    traceback=[f"{exc.__class__.__name__}: {exc}"],
                )
            ]
            cell.execution_count = execution_count
            output_path.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(nb, output_path)
            raise

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    execute_notebook(Path(args.input), Path(args.output))
    print(f"Executed notebook written to {args.output}")


if __name__ == "__main__":
    main()
