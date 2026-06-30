"""Custom exceptions for ml_package.

Keep error taxonomy explicit so API/CLI can map failures to clear messages.
"""


class MLPackageError(Exception):
    """Base exception for all package-level errors."""


class ModelNotLoadedError(MLPackageError):
    """Raised when model access is attempted before loading."""


class ArtifactVerificationError(MLPackageError):
    """Raised when serialized artifact integrity checks fail."""


class UnsafeDeserializationError(MLPackageError):
    """Raised when unsafe deserialization is blocked by configuration."""
