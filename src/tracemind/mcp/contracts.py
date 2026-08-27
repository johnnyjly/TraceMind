"""Client reply validation and deprecated stdio argument definitions."""

from typing import Annotated

from pydantic import (
    AwareDatetime, BaseModel, ConfigDict, Field, model_validator,
)

from tracemind.models import Evidence


class Arguments(BaseModel):
    """Reject fields outside a registered telemetry operation."""

    model_config = ConfigDict(extra="forbid")


class Window(Arguments):
    """Carry the application-selected timezone-aware investigation window."""

    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def validate_window(self) -> "Window":
        """Require an ordered, nonempty query window."""
        if self.end <= self.start:
            raise ValueError("end must be after start")
        return self


class MetricQuery(Arguments):
    """Query an instant vector at the application-selected end time."""

    expression: str
    at: AwareDatetime


class MetricDiscovery(Window):
    """Select a bounded metric-name or label catalogue."""

    service: str | None = Field(default=None, max_length=120)
    metric: str | None = Field(default=None, max_length=240)
    limit: int = Field(default=20, ge=1, le=50)


class TraceSearch(Window):
    """Select one service's trace summaries."""

    service: str


class TraceDetail(Arguments):
    """Require an actual Jaeger trace ID rather than an evidence UUID."""

    trace_id: Annotated[str, Field(
        pattern=r"^(?:[0-9a-fA-F]{16}|[0-9a-fA-F]{32})$",
    )]


class LogSearch(TraceSearch):
    """Select service logs with an optional body-text filter."""

    text: str | None = None


TOOL_ARGUMENTS = {
    "query_metrics": MetricQuery,
    "discover_metrics": MetricDiscovery,
    "search_traces": TraceSearch,
    "get_trace_detail": TraceDetail,
    "search_logs": LogSearch,
}


class RemoteFailure(BaseModel):
    """Transfer safe retry classification, excluding backend error text."""

    model_config = ConfigDict(extra="forbid")
    category: str
    retryable: bool = False
    status_code: int | None = None
    retry_after: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class TelemetryReply(BaseModel):
    """Return exactly one observation list, metadata object or safe failure."""

    model_config = ConfigDict(extra="forbid")
    evidence: list[Evidence] | None = None
    metadata: dict | None = None
    error: RemoteFailure | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "TelemetryReply":
        """Prevent metadata or failure records from becoming observations."""
        if sum(item is not None for item in (
            self.evidence, self.metadata, self.error,
        )) != 1:
            raise ValueError("reply requires exactly one payload")
        return self
