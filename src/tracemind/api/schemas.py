"""Validated HTTP request contracts for investigation windows."""

from datetime import datetime
from uuid import UUID

from typing import Literal

from pydantic import AwareDatetime, BaseModel, Field, model_validator

from tracemind.models import Diagnosis, Incident
from tracemind.models.run import RunPurpose, RunStatus, StopReason


class DiagnosisRequest(BaseModel):
    """Incident and timezone-aware start/end bounds for an investigation."""

    incident: Incident
    investigation_start: AwareDatetime
    investigation_end: AwareDatetime

    @model_validator(mode="after")
    def validate_window(self) -> "DiagnosisRequest":
        """Reject windows whose end is not strictly after their start."""
        if self.investigation_end <= self.investigation_start:
            raise ValueError(
                "investigation_end must be after investigation_start"
            )
        return self


class RunStatusResponse(BaseModel):
    """Bounded lifecycle view for a persistent investigation."""

    id: UUID
    investigation_id: UUID | None = None
    parent_run_id: UUID | None = None
    purpose: RunPurpose = RunPurpose.AUTOMATIC
    status: RunStatus
    stop_reason: StopReason | None
    cancel_requested: bool
    attempt: int
    model_calls: int
    tool_calls: int
    evidence_count: int
    started_at: datetime | None
    finished_at: datetime | None


class FollowUpRequest(BaseModel):
    """An optional human question with an explicit execution policy."""

    message: str = Field(min_length=1, max_length=4000)
    purpose: Literal["investigate", "explain"] = "investigate"
    investigation_start: AwareDatetime | None = None
    investigation_end: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_request(self) -> "FollowUpRequest":
        """Require nonblank input and a complete, increasing time window."""
        if not self.message.strip():
            raise ValueError("message must not be blank")
        if ((self.investigation_start is None) !=
                (self.investigation_end is None)):
            raise ValueError("provide both investigation bounds or neither")
        if (self.investigation_start is not None and
                self.investigation_end <= self.investigation_start):
            raise ValueError("investigation_end must be after start")
        return self


class InvestigationCreated(BaseModel):
    """Created incident record and its autonomous first round."""

    id: UUID
    run: RunStatusResponse


class InvestigationTurn(BaseModel):
    """User request and the preserved outcome of an investigation round."""

    run_id: UUID
    parent_run_id: UUID | None
    purpose: RunPurpose
    status: RunStatus
    stop_reason: StopReason | None
    investigation_start: datetime
    investigation_end: datetime
    user_message: str | None
    diagnosis: Diagnosis | None


class InvestigationResponse(BaseModel):
    """Incident metadata with a bounded, chronological round history."""

    id: UUID
    incident: Incident
    created_at: datetime
    turns: list[InvestigationTurn]
    next_offset: int | None


class ProgressEvent(BaseModel):
    """Readable progress with allowlisted metadata, separate from raw logs."""

    run_id: UUID
    span_id: UUID
    parent_span_id: UUID | None
    sequence: int
    timestamp: datetime
    kind: str
    phase: str
    message: str
    data: dict = Field(default_factory=dict)
