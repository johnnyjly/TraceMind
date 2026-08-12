"""Failures reported by telemetry transports and response decoders."""


class TelemetryClientError(RuntimeError):
    """Raised when a telemetry backend cannot return usable evidence."""
