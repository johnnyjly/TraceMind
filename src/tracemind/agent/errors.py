class ModelConfigurationError(ValueError):
    """Raised when the selected model provider is not configured."""


class DiagnosisError(RuntimeError):
    """Raised when an agent result violates the diagnosis contract."""
