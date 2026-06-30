"""Optional OpenVINO export helper."""

from __future__ import annotations

from pathlib import Path


def openvino_available() -> bool:
    """Check whether OpenVINO runtime is installed."""
    try:
        import openvino  # noqa: F401
    except Exception:
        return False
    return True


def export_openvino_placeholder(output_dir: Path) -> Path:
    """Write marker artifact when OpenVINO path enabled in config."""
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "openvino_status.txt"
    marker.write_text("OpenVINO export path placeholder. Install optional dependency.", encoding="utf-8")
    return marker
