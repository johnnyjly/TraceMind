"""Validated HTTP request contracts for investigation windows."""

from pydantic import AwareDatetime, BaseModel, model_validator

from tracemind.models import Incident


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
