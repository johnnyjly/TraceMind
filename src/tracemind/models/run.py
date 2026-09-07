"""Run lifecycle, execution limits, operational events, and token usage."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, StringConstraints

from tracemind.models.diagnosis import Diagnosis
from tracemind.models.incident import Incident
from tracemind.models.verification import (
    VerificationPolicy, VerificationRecord,
)
from tracemind.models.knowledge import KnowledgePolicy, KnowledgeReference
from tracemind.models.memory import (
    FeedbackUpdate,
    GuidanceSnapshot,
    MemoryPolicy,
    MemorySnapshot,
)


ToolName = Annotated[str, StringConstraints(
    min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$",
)]
# Default grants for the Demo, rather than an exhaustive tool catalogue.
TOOL_NAMES: tuple[ToolName, ...] = (
    "query_metrics", "search_traces", "get_trace_detail", "search_logs",
    "read_evidence", "discover_metrics"
)


class RunLimits(BaseModel):
    """Call, time, context, output, and evidence-size limits for one run."""

    context_tokens: int = Field(default=32768, ge=1)
    output_tokens: int = Field(default=4096, ge=1)
    observation_bytes: int = Field(default=8_388_608, ge=1)
    model_calls: int = Field(default=8, ge=1)
    tool_calls: int = Field(default=12, ge=1)
    identical_tool_calls: int = Field(default=2, ge=1)
    duration_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    model_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    tool_timeout_seconds: float = Field(default=20, gt=0, allow_inf_nan=False)
    retry_attempts: int = Field(default=2, ge=0, le=5)
    retry_backoff_seconds: float = Field(
        default=0.25, ge=0, allow_inf_nan=False
    )
    retry_max_backoff_seconds: float = Field(
        default=4, ge=0, allow_inf_nan=False
    )
    diagnosis_repairs: int = Field(default=1, ge=0, le=3)
    no_progress_rounds: int = Field(default=3, ge=1)


class RunStatus(StrEnum):
    """Execution lifecycle status, separate from diagnostic certainty."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    INCONCLUSIVE = "inconclusive"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunPurpose(StrEnum):
    """Distinguish autonomous diagnosis, follow-up checks and explanation."""

    AUTOMATIC = "automatic"
    INVESTIGATE = "investigate"
    EXPLAIN = "explain"


class StopReason(StrEnum):
    """Machine-readable reason why an investigation reached a terminal
    state.
    """

    COMPLETED = "completed"
    VERIFICATION_REJECTED = "verification_rejected"
    VERIFICATION_FAILED = "verification_failed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONTEXT_LIMIT = "context_limit"
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
    NO_PROGRESS = "no_progress"


class RunEvent(BaseModel):
    """Ordered operational metadata for a run and its related spans."""

    run_id: UUID
    sequence: int
    kind: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    span_id: UUID
    parent_span_id: UUID | None = None
    duration_ms: float | None = None
    # Allowlisted metadata and bounded rejected diagnosis excerpts only;
    # never full prompts or raw tool logs.
    data: dict[str, str | int | float | bool | list[str] | None] = Field(
        default_factory=dict
    )


class TokenUsage(BaseModel):
    """Provider-reported totals and counts of calls with missing usage."""

    reported_model_calls: int = 0
    unreported_model_calls: int = 0
    reported_input_tokens: int = 0
    reported_output_tokens: int = 0
    reported_total_tokens: int = 0


class RunState(BaseModel):
    """Mutable investigation state, counters, evidence IDs, and events.
    """

    id: UUID = Field(default_factory=uuid4)
    investigation_id: UUID | None = None
    parent_run_id: UUID | None = None
    purpose: RunPurpose = RunPurpose.AUTOMATIC
    user_message: str | None = Field(default=None, max_length=4000)
    previous_messages: list[str] = Field(default_factory=list)
    prior_diagnosis: Diagnosis | None = None
    omitted_prior_evidence: int = 0
    memory_policy: MemoryPolicy | None = None
    memory_snapshot: MemorySnapshot | None = None
    learning_policy: MemoryPolicy | None = None
    guidance_snapshot: GuidanceSnapshot | None = None
    feedback_updates: list[FeedbackUpdate] = Field(
        default_factory=list, max_length=3,
    )
    incident: Incident
    investigation_start: datetime
    investigation_end: datetime
    limits: RunLimits
    allowed_tools: tuple[ToolName, ...]
    knowledge_policy: KnowledgePolicy | None = None
    knowledge: list[KnowledgeReference] = Field(default_factory=list)
    prompt_version: str = "single-agent-v7"
    telemetry_transport: Literal["direct", "mcp", "replay"] = "direct"
    status: RunStatus = RunStatus.RUNNING
    stop_reason: StopReason | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deadline_at: datetime | None = None
    finished_at: datetime | None = None
    elapsed_ms: float | None = None
    model_calls: int = 0
    backend_attempts: int = 0
    diagnosis_repairs: int = 0
    stalled_rounds: int = 0
    model_switches: int = 0
    active_model: str | None = None
    tool_calls: dict[str, int] = Field(default_factory=dict)
    tool_signatures: dict[str, int] = Field(default_factory=dict)
    result_fingerprints: list[str] = Field(default_factory=list)
    progress_count: int = 0
    last_progress: int | None = None
    evidence_ids: list[UUID] = Field(default_factory=list)
    diagnosis: Diagnosis | None = None
    verification_policy: VerificationPolicy | None = None
    verification: VerificationRecord | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    events: list[RunEvent] = Field(default_factory=list)
