"""Bounded, reported investigation hints; never retained telemetry evidence."""

import json
import re

from pydantic import (
    AwareDatetime, BaseModel, ConfigDict, Field, field_validator,
    model_validator,
)


MAX_INITIAL_CONTEXT_BYTES = 32768
_CREDENTIAL = re.compile(
    r"(?i)(?<![\w])([\"']?(?:password|passwd|pwd|token|access[_-]?token|"
    r"refresh[_-]?token|client[_-]?secret|secret|api[_-]?key|authorization|"
    r"proxy-authorization|cookie|set-cookie)[\"']?\s*[:=]\s*)"
    r"(?:\[REDACTED\]|\"[^\"]*\"|'[^']*'|[^\s,;&}\]]+)"
)
_AUTHORIZATION = re.compile(
    r"(?i)(\b(?:Bearer|Basic)\s+)[A-Za-z0-9._~+/=-]+"
)
_HEADER = re.compile(
    r"(?im)(\b(?:authorization|proxy-authorization|cookie|set-cookie)"
    r"\s*:\s*)[^\r\n]+"
)
_URL_CREDENTIAL = re.compile(
    r"(?i)([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@"
)
_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----[\s\S]*?"
    r"-----END (?:[A-Z ]+ )?PRIVATE KEY-----"
)


def redact_reported_text(value: str, *, preserve_lines: bool = False) -> str:
    """Redact known credential forms, optionally retaining line numbers."""
    for pattern, replacement in (
        (_PRIVATE_KEY, "[REDACTED PRIVATE KEY]"),
        (_URL_CREDENTIAL, r"\1[REDACTED]@"),
        (_HEADER, r"\1[REDACTED]"),
        (_AUTHORIZATION, r"\1[REDACTED]"),
        (_CREDENTIAL, r"\1[REDACTED]"),
    ):

        def replace(match):
            cleaned = match.expand(replacement)
            if preserve_lines:
                missing = match.group().count("\n") - cleaned.count("\n")
                cleaned += "\n[REDACTED]" * max(0, missing)
            return cleaned

        value = pattern.sub(replace, value)
    return value


class ReportedContext(BaseModel):
    """Reject unknown fields and sanitize text before state storage."""

    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, hide_input_in_errors=True,
    )

    @field_validator("*", mode="before")
    @classmethod
    def sanitize_text(cls, value):
        return redact_reported_text(value) if isinstance(value, str) else value


class ContextOrigin(ReportedContext):
    """Reporter-supplied provenance, not independent source verification."""

    source: str | None = Field(default=None, min_length=1, max_length=120)
    external_id: str | None = Field(default=None, min_length=1, max_length=256)
    reported_at: AwareDatetime | None = None


class TraceReference(ReportedContext):
    """Known identifiers select a trace; service and operation remain hints."""

    trace_id: str = Field(pattern=r"^(?:[0-9a-f]{16}|[0-9a-f]{32})$")
    span_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")
    service: str | None = Field(default=None, min_length=1, max_length=120)
    operation: str | None = Field(default=None, min_length=1, max_length=240)

    @field_validator("trace_id", "span_id", mode="before")
    @classmethod
    def normalize_identifier(cls, value):
        if isinstance(value, str):
            value = value.strip().lower()
            if value and set(value) == {"0"}:
                raise ValueError("trace and span identifiers must be nonzero")
        return value


class ReportedError(ReportedContext):
    """A supplied exception and location, not a verified code reference."""

    type: str | None = Field(default=None, min_length=1, max_length=240)
    message: str | None = Field(default=None, min_length=1, max_length=4096)
    stacktrace: str | None = Field(
        default=None, min_length=1, max_length=16384,
    )
    file: str | None = Field(default=None, min_length=1, max_length=1024)
    line: int | None = Field(default=None, ge=1, strict=True)


class ContextScope(ReportedContext):
    """Reported runtime scope; it cannot change configured backend access."""

    environment: str | None = Field(default=None, min_length=1, max_length=120)
    cluster: str | None = Field(default=None, min_length=1, max_length=120)
    namespace: str | None = Field(default=None, min_length=1, max_length=120)


class DeploymentHint(ReportedContext):
    """A service-specific deployment claim to corroborate with observations."""

    service: str = Field(min_length=1, max_length=120)
    version: str | None = Field(default=None, min_length=1, max_length=240)
    instance: str | None = Field(default=None, min_length=1, max_length=240)


class InitialContext(ReportedContext):
    """Per-run hints, separate from Incident and retained evidence."""

    prefetch_traces: bool = True
    origin: ContextOrigin | None = None
    traces: list[TraceReference] = Field(default_factory=list, max_length=3)
    error: ReportedError | None = None
    scope: ContextScope | None = None
    deployments: list[DeploymentHint] = Field(
        default_factory=list, max_length=10,
    )

    @model_validator(mode="before")
    @classmethod
    def bound_input(cls, value):
        if isinstance(value, dict):
            size = len(json.dumps(value, ensure_ascii=False,
                                  default=str).encode())
            if size > MAX_INITIAL_CONTEXT_BYTES:
                raise ValueError("initial context exceeds 32768 bytes")
        return value

    @model_validator(mode="after")
    def bound_serialized_context(self):
        if len(self.model_dump_json().encode()) > MAX_INITIAL_CONTEXT_BYTES:
            raise ValueError("initial context exceeds 32768 bytes")
        return self
