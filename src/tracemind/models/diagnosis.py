"""Structured diagnostic conclusions and uncertainty status."""

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class DiagnosisStatus(StrEnum):
    """Whether the diagnostic result is complete or inconclusive."""

    COMPLETE = "complete"
    INCONCLUSIVE = "inconclusive"


class Diagnosis(BaseModel):
    """Incident-linked conclusion with confidence, citations, and
    remediation.
    """

    incident_id: UUID
    status: DiagnosisStatus
    root_cause: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[UUID] = Field(default_factory=list)
    knowledge_refs: list[str] = Field(default_factory=list)
    remediation: list[str] = Field(default_factory=list)
