from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class DiagnosisStatus(StrEnum):
    COMPLETE = "complete"
    INCONCLUSIVE = "inconclusive"


class Diagnosis(BaseModel):
    incident_id: UUID
    status: DiagnosisStatus
    root_cause: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[UUID] = Field(default_factory=list)
    remediation: list[str] = Field(default_factory=list)
