"""Service-owned MCP schemas for tool discovery and structured replies."""

from typing import Annotated

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from otel_telemetry_mcp.observations import Evidence


class Arguments(BaseModel):
    """Reject fields outside a registered telemetry operation."""

    model_config = ConfigDict(extra="forbid")


class Window(Arguments):
    """Carry the application-selected timezone-aware investigation window."""

    start: AwareDatetime = Field(
        json_schema_extra={"x-investigation-context": "start"}
    )
    end: AwareDatetime = Field(
        json_schema_extra={"x-investigation-context": "end"}
    )

    @model_validator(mode="after")
    def validate_window(self) -> "Window":
        """Require an ordered, nonempty query window."""
        if self.end <= self.start:
            raise ValueError("end must be after start")
        return self


class MetricQuery(Arguments):
    """Query an instant vector at the application-selected end time."""

    expression: str
    at: AwareDatetime = Field(
        json_schema_extra={"x-investigation-context": "end"}
    )


class MetricDiscovery(Window):
    """Select a bounded metric-name or label catalogue."""

    service: str | None = Field(default=None, max_length=120)
    metric: str | None = Field(default=None, max_length=240)
    limit: int = Field(default=20, ge=1, le=50)


class TraceSearch(Window):
    """Select one service's trace summaries."""

    service: str
    operation: str | None = Field(default=None, max_length=240)
    min_duration_ms: float | None = Field(
        default=None, gt=0, allow_inf_nan=False
    )
    errors_only: bool = False


class TraceDetail(Arguments):
    """Require an actual Jaeger trace ID rather than an evidence UUID."""

    trace_id: Annotated[
        str,
        Field(
            pattern=r"^(?:[0-9a-fA-F]{16}|[0-9a-fA-F]{32})$",
        ),
    ]
    service: str | None = Field(default=None, max_length=120)
    operation: str | None = Field(default=None, max_length=240)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=25, ge=1, le=100)


class LogSearch(Window):
    """Select service logs with an optional body-text filter."""

    service: str
    text: str | None = None
    trace_id: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{16,32}$")


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
        if (
            sum(
                item is not None
                for item in (
                    self.evidence,
                    self.metadata,
                    self.error,
                )
            )
            != 1
        ):
            raise ValueError("reply requires exactly one payload")
        return self
