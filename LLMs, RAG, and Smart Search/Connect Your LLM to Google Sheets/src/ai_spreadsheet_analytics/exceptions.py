"""Custom exceptions."""


class AnalyticsError(Exception):
    """Base analytics exception."""


class CredentialError(AnalyticsError):
    """Credential or auth error."""


class DataValidationError(AnalyticsError):
    """Raised when data validation fails."""


class LLMError(AnalyticsError):
    """Raised when LLM request fails."""


class ReportError(AnalyticsError):
    """Raised when report generation fails."""
