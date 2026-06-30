"""Capture local validation evidence for deployment documentation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "deployment"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    pytest_run = subprocess.run(
        ["uv", "run", "pytest", "-q"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    screenshot_paths = sorted((ROOT / "outputs" / "screenshots").glob("*.png"))
    screenshots = [str(p.relative_to(ROOT)) for p in screenshot_paths]
    diagrams = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "outputs" / "figures").glob("*.png"))
    executed_nbs = sorted(
        str(p.relative_to(ROOT))
        for p in (ROOT / "outputs" / "executed_notebooks").rglob("*.ipynb")
    )

    seen_hashes: dict[str, str] = {}
    duplicate_screenshots: list[str] = []
    for path in screenshot_paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rel = str(path.relative_to(ROOT))
        if digest in seen_hashes:
            duplicate_screenshots.append(rel)
        else:
            seen_hashes[digest] = rel

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pytest_exit_code": pytest_run.returncode,
        "pytest_tail": (pytest_run.stdout + "\n" + pytest_run.stderr).splitlines()[-10:],
        "screenshot_count": len(screenshots),
        "unique_screenshot_count": len(seen_hashes),
        "duplicate_screenshots": duplicate_screenshots,
        "screenshots": screenshots,
        "diagram_count": len(diagrams),
        "executed_notebook_count": len(executed_nbs),
        "executed_notebooks": executed_nbs,
        "notes": "Cloud deployment URL field should be filled after authenticated Streamlit Cloud release.",
    }

    out_path = OUT_DIR / "local_validation_evidence.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved evidence: {out_path}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
