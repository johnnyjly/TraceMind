"""Structured diagnostic conclusions and uncertainty status."""

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class DiagnosisStatus(StrEnum):
    """Whether the diagnostic result is complete or inconclusive."""

    COMPLETE = "complete"
    INCONCLUSIVE = "inconclusive"


class TriageItem(BaseModel):
    """A retained statement with review and evidence support kept separate."""

    id: str
    section: Literal["finding", "hypothesis", "limitation", "proposed_check"]
    text: str = Field(min_length=1)
    evidence_ids: list[UUID] = Field(default_factory=list)
    knowledge_refs: list[str] = Field(default_factory=list)
    review_status: Literal["reviewed", "unreviewed"]
    kind: Literal[
        "diagnostic", "uncertainty", "process", "proposal"
    ] | None = None
    evidence_support: Literal[
        "supported", "insufficient", "contradicted", "not_assessed",
        "not_applicable",
    ]
    supporting_evidence_ids: list[UUID] = Field(default_factory=list)
    conflicting_evidence_ids: list[UUID] = Field(default_factory=list)
    supporting_event_sequences: list[int] = Field(default_factory=list)
    conflicting_event_sequences: list[int] = Field(default_factory=list)
    missing_observations: list[str] = Field(default_factory=list)
    rationale: str | None = None


class Diagnosis(BaseModel):
    """Incident-linked conclusion with confidence, citations, and
    remediation.
    """

    incident_id: UUID
    status: DiagnosisStatus
    root_cause: str = Field(min_length=1)
    confidence: float = Field(
        ge=0, le=1,
        description="Legacy uncalibrated model estimate, not a probability.",
    )
    evidence_ids: list[UUID] = Field(default_factory=list)
    knowledge_refs: list[str] = Field(default_factory=list)
    remediation: list[str] = Field(default_factory=list)
    items: list[TriageItem] = Field(default_factory=list)
    review_status: Literal[
        "not_requested", "completed", "timed_out", "unavailable"
    ] = "not_requested"
    review_error_code: str | None = None
    triage_status: Literal["ready", "partial", "no_findings"] | None = None
