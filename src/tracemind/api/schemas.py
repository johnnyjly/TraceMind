"""Validated HTTP request contracts for investigation windows."""

from datetime import datetime
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, model_validator

from tracemind.models import Incident
from tracemind.models.run import RunStatus, StopReason


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
    status: RunStatus
    stop_reason: StopReason | None
    cancel_requested: bool
    attempt: int
    model_calls: int
    tool_calls: int
    evidence_count: int
    started_at: datetime | None
    finished_at: datetime | None
