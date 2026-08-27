"""Failures reported by telemetry transports and response decoders."""


class TelemetryClientError(RuntimeError):
    """Raised when a telemetry backend cannot return usable evidence."""


class RemoteTelemetryError(TelemetryClientError):
    """Preserve safe remote classification without remote exception text."""

    def __init__(
        self, category: str, *, retryable: bool = False,
        status_code: int | None = None, retry_after: float | None = None,
    ) -> None:
        """Retain the fields required by the harness's retry policy."""
        super().__init__(category)
        self.category = category
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after = retry_after
