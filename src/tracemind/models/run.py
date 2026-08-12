from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from tracemind.models.incident import Incident


ToolName = Literal[
    "query_metrics", "search_traces", "get_trace_detail", "search_logs"
]
TOOL_NAMES: tuple[ToolName, ...] = (
    "query_metrics", "search_traces", "get_trace_detail", "search_logs"
)


class RunLimits(BaseModel):
    model_calls: int = Field(default=8, ge=1)
    tool_calls: int = Field(default=12, ge=1)
    identical_tool_calls: int = Field(default=2, ge=1)
    duration_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    model_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    tool_timeout_seconds: float = Field(default=20, gt=0, allow_inf_nan=False)


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    INCONCLUSIVE = "inconclusive"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StopReason(StrEnum):
    COMPLETED = "completed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    MODEL_BUDGET = "model_budget"
    TOOL_BUDGET = "tool_budget"
    REPEATED_TOOL_CALL = "repeated_tool_call"
    DEADLINE = "deadline"
    MODEL_TIMEOUT = "model_timeout"
    TOOL_TIMEOUT = "tool_timeout"
    RECURSION_LIMIT = "recursion_limit"
    TOOL_DENIED = "tool_denied"
    TOOL_ERROR = "tool_error"
    MODEL_ERROR = "model_error"
    INVALID_DIAGNOSIS = "invalid_diagnosis"
    INTERNAL_ERROR = "internal_error"
    CANCELLED = "cancelled"


class RunEvent(BaseModel):
    run_id: UUID
    sequence: int
    kind: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    span_id: UUID
    parent_span_id: UUID | None = None
    duration_ms: float | None = None
    # Only allowlisted operational metadata belongs here,
    # never prompts or logs.
    data: dict[str, str | int | float | bool | list[str] | None] = Field(
        default_factory=dict
    )


class TokenUsage(BaseModel):
    reported_model_calls: int = 0
    unreported_model_calls: int = 0
    reported_input_tokens: int = 0
    reported_output_tokens: int = 0
    reported_total_tokens: int = 0


class RunState(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    incident: Incident
    investigation_start: datetime
    investigation_end: datetime
    limits: RunLimits
    allowed_tools: tuple[ToolName, ...]
    prompt_version: str = "single-agent-v2"
    status: RunStatus = RunStatus.RUNNING
    stop_reason: StopReason | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    elapsed_ms: float | None = None
    model_calls: int = 0
    tool_calls: dict[str, int] = Field(default_factory=dict)
    evidence_ids: list[UUID] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    events: list[RunEvent] = Field(default_factory=list)
