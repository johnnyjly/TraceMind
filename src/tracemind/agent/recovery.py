"""Classify service failures without exposing provider or backend payloads."""

from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from math import isfinite

import httpx
from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
)
from openai import APIConnectionError

from tracemind.telemetry.errors import RemoteTelemetryError


@dataclass(frozen=True)
class Failure:
    """Safe error classification and an optional server backoff hint."""

    category: str
    retryable: bool = False
    status_code: int | None = None
    retry_after: float | None = None


def retry_after_seconds(value: str | None) -> float | None:
    """Parse Retry-After seconds or an HTTP date; reject invalid values."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            seconds = (date - datetime.now(UTC)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0, seconds) if isfinite(seconds) else None


def classify_failure(error: Exception) -> Failure:
    """Inspect typed errors and their causes; never classify message text."""
    chain = []
    current = error
    while current is not None and id(current) not in {id(e) for e in chain}:
        chain.append(current)
        current = current.__cause__ or current.__context__
    for item in reversed(chain):
        if isinstance(item, RemoteTelemetryError):
            return Failure(item.category, item.retryable,
                           item.status_code, item.retry_after)
        response = getattr(item, "response", None)
        status = getattr(item, "status_code", None)
        if status is None and response is not None:
            status = getattr(response, "status_code", None)
        if status is None:
            status = getattr(item, "code", None)
        if isinstance(status, int):
            headers = getattr(response, "headers", {})
            hint = retry_after_seconds(headers.get("retry-after"))
            if status in {408, 429, 500, 502, 503, 504}:
                return Failure("transient", True, status, hint)
            if status in {401, 403}:
                return Failure("access_denied", status_code=status)
            if 400 <= status < 500:
                return Failure("invalid_request", status_code=status)
            return Failure("service_error", status_code=status)
        if isinstance(item, (ModelAuthenticationError,
                             ModelPermissionDeniedError)):
            return Failure("access_denied")
        if isinstance(item, (ModelInvalidRequestError, ModelNotFoundError)):
            return Failure("invalid_request")
        if isinstance(item, (TimeoutError, httpx.TransportError,
                             APIConnectionError)):
            return Failure("transient", True)
        if isinstance(item, ModelError) and item.is_retryable:
            return Failure("transient", True)
    return Failure("malformed_or_unclassified")


class ModelCallFailed(RuntimeError):
    """Expose an exhausted transient failure to the fallback workflow."""

    def __init__(self, failure: Failure, *, timed_out: bool) -> None:
        """Retain safe classification without copying exception messages."""
        super().__init__(failure.category)
        self.failure = failure
        self.timed_out = timed_out


class ToolUnavailable(RuntimeError):
    """Report a recoverable query failure to the diagnostic agent."""

    def __init__(self, failure: Failure) -> None:
        """Retain safe feedback fields for the tool response."""
        super().__init__(failure.category)
        self.failure = failure
