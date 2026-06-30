"""Advanced alignment workflow stubs (DPO/ORPO/GRPO)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llmft.utils.io import write_json


@dataclass(slots=True)
class AlignmentRequest:
    """Alignment run request."""

    method: str
    dataset_path: str
    model_id: str


class AlignmentEngine:
    """Config-integrated alignment stubs for future expansion."""

    SUPPORTED = {"dpo", "orpo", "grpo"}

    def run(self, request: AlignmentRequest, artifacts_dir: str | Path) -> Path:
        """Emit alignment stub artifact.

        This v1 implementation provides execution contract and metadata only.
        """
        method = request.method.lower()
        if method not in self.SUPPORTED:
            raise ValueError(f"Unsupported alignment method: {request.method}")

        path = Path(artifacts_dir) / "training" / f"alignment-{method}.json"
        write_json(
            path,
            {
                "method": method,
                "dataset_path": request.dataset_path,
                "model_id": request.model_id,
                "status": "stubbed_v1",
                "next": "wire TRL preference trainer when compute/data budget allows",
            },
        )
        return path
