"""Translate HTTP failures into safe, transport-independent reply fields."""

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from math import isfinite

import httpx

from otel_telemetry_mcp.contracts import RemoteFailure


def classify_failure(error: Exception) -> RemoteFailure:
    """Inspect causes without transferring exception text or backend bodies."""
    visited = set()
    current = error
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, httpx.HTTPStatusError):
            status = current.response.status_code
            retryable = status in {408, 429, 500, 502, 503, 504}
            category = ("transient" if retryable else "access_denied"
                        if status in {401, 403} else "not_found"
                        if status == 404 else "invalid_request"
                        if 400 <= status < 500 else "service_error")
            hint = retry_after(current.response.headers.get("retry-after"))
            return RemoteFailure(category=category, retryable=retryable,
                                 status_code=status, retry_after=hint)
        if isinstance(current, (TimeoutError, httpx.TransportError)):
            return RemoteFailure(category="transient", retryable=True)
        current = current.__cause__ or current.__context__
    return RemoteFailure(category="malformed_or_unclassified")


def retry_after(value: str | None) -> float | None:
    """Parse a backend Retry-After hint without retaining raw header values."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (parsedate_to_datetime(value)
                       - datetime.now(UTC)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0, seconds) if isfinite(seconds) else None
