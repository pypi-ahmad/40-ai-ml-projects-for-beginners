import pickle
from pathlib import Path
from typing import Any

import joblib

from ml_package.artifact_security import (
    compute_sha256,
    manifest_path_for,
    read_manifest,
    verify_manifest_checksum,
    write_manifest,
)
from ml_package.exceptions import ArtifactVerificationError, UnsafeDeserializationError


class ModelLoader:
    """Unified model loader with optional integrity and trust checks."""

    SUPPORTED_FORMATS = {".pkl", ".joblib", ".onnx", ".pt"}
    UNSAFE_FORMATS = {".pkl", ".joblib"}

    def __init__(
        self,
        model_path: str | Path,
        *,
        verify_integrity: bool = False,
        require_manifest: bool = False,
        trusted_digests: set[str] | None = None,
        allow_unsafe_deserialization: bool = True,
    ):
        self.model_path = Path(model_path)
        self.format = self._infer_format(self.model_path)
        self.verify_integrity = verify_integrity
        self.require_manifest = require_manifest
        self.trusted_digests = trusted_digests or set()
        self.allow_unsafe_deserialization = allow_unsafe_deserialization
        self._model: Any = None
        self._last_digest: str | None = None

    def _infer_format(self, path: Path) -> str:
        file_format = path.suffix.lower()
        if file_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{file_format}'. "
                f"Supported: {sorted(self.SUPPORTED_FORMATS)}"
            )
        return file_format

    def _verify_digest_policy(self, model_path: Path) -> str:
        digest = compute_sha256(model_path)

        if self.trusted_digests and digest not in self.trusted_digests:
            raise ArtifactVerificationError(
                "Artifact digest not in trusted digest allow-list. "
                f"digest={digest}, path={model_path}"
            )

        manifest_exists = manifest_path_for(model_path).exists()
        if self.verify_integrity or manifest_exists:
            verify_manifest_checksum(
                model_path,
                required=self.require_manifest or self.verify_integrity,
            )
        elif self.require_manifest:
            raise ArtifactVerificationError(
                f"Manifest required but missing for artifact: {model_path}"
            )

        return digest

    def load(self) -> Any:
        """Load model from disk using deserializer matching file extension."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self._last_digest = self._verify_digest_policy(self.model_path)

        if self.format in self.UNSAFE_FORMATS and not self.allow_unsafe_deserialization:
            if self._last_digest not in self.trusted_digests:
                raise UnsafeDeserializationError(
                    "Unsafe deserialization blocked for pickle/joblib. "
                    "Only artifacts whose digest is in trusted_digests are allowed when "
                    "allow_unsafe_deserialization=False. "
                    f"digest={self._last_digest}"
                )

        if self.format == ".joblib":
            self._model = joblib.load(self.model_path)
        elif self.format == ".pkl":
            with self.model_path.open("rb") as handle:
                self._model = pickle.load(handle)
        elif self.format == ".onnx":
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError(
                    "onnxruntime is required to load .onnx models. "
                    "Install with: uv add onnxruntime"
                ) from exc
            self._model = ort.InferenceSession(str(self.model_path))
        elif self.format == ".pt":
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "torch is required to load .pt artifacts. "
                    "Install with: uv add torch"
                ) from exc
            self._model = torch.jit.load(str(self.model_path))

        return self._model

    def save(
        self,
        model: Any,
        path: str | Path | None = None,
        *,
        create_manifest: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Serialize model to disk using format inferred from target path."""
        save_path = Path(path) if path else self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_format = self._infer_format(save_path)

        if save_format == ".joblib":
            joblib.dump(model, save_path)
        elif save_format == ".pkl":
            with save_path.open("wb") as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        elif save_format == ".onnx":
            try:
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
            except ImportError as exc:
                raise RuntimeError(
                    "skl2onnx is required to save .onnx models. "
                    "Install with: uv add skl2onnx"
                ) from exc

            n_features = int(getattr(model, "n_features_in_", 4))
            initial_type = [("float_input", FloatTensorType([None, n_features]))]
            onx = convert_sklearn(model, initial_types=initial_type)
            with save_path.open("wb") as handle:
                handle.write(onx.SerializeToString())
        elif save_format == ".pt":
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "torch is required to save .pt artifacts. "
                    "Install with: uv add torch"
                ) from exc

            if hasattr(model, "save") and "torch" in type(model).__module__:
                model.save(str(save_path))
            else:
                raise TypeError(
                    "Saving .pt requires a torch.jit.ScriptModule or compatible torch object"
                )

        if create_manifest:
            write_manifest(save_path, metadata=metadata)

        self.model_path = save_path
        self.format = save_format
        self._last_digest = compute_sha256(save_path)
        return save_path

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata about model path and loading status."""
        manifest = read_manifest(self.model_path)
        return {
            "path": str(self.model_path),
            "format": self.format,
            "size_bytes": self.model_path.stat().st_size if self.model_path.exists() else None,
            "is_loaded": self._model is not None,
            "manifest_exists": manifest is not None,
            "sha256": self._last_digest,
        }

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")
        return self._model
