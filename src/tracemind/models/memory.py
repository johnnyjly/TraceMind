"""Human-confirmed incident experience, separate from runtime evidence."""

import hashlib
import json
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints,
    model_validator,
)


ScopeName = Annotated[str, StringConstraints(
    strip_whitespace=True, min_length=1, max_length=120,
)]
QueryText = Annotated[str, StringConstraints(
    strip_whitespace=True, min_length=1, max_length=400,
)]


class MemoryScope(BaseModel):
    """Exact deployment identity; service names alone are insufficient."""

    model_config = ConfigDict(extra="forbid")
    environment: ScopeName
    deployment_version: ScopeName


class MemoryPolicy(BaseModel):
    """Capture opt-in retrieval scope and bounds when a run is created."""

    scope: MemoryScope
    max_records: int = Field(default=3, ge=1, le=10)
    context_bytes: int = Field(default=4096, ge=512, le=16000)
    auto_expiry_days: int = Field(default=30, ge=1, le=365)


class MemoryContent(MemoryScope):
    """Explicit human confirmation, not a copied model-confidence score."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    services: list[ScopeName] = Field(min_length=1, max_length=20)
    symptom: str = Field(min_length=1, max_length=1000)
    resolution: str = Field(min_length=1, max_length=2000)
    diagnostic_queries: list[QueryText] = Field(
        default_factory=list, max_length=5,
    )
    remediation: list[QueryText] = Field(default_factory=list, max_length=5)
    confirmed_by: ScopeName
    confirmation_note: str = Field(min_length=1, max_length=1000)
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def unique_services(self) -> "MemoryContent":
        """Keep service matching exact and reject ambiguous duplicate scope."""
        if len(set(self.services)) != len(self.services):
            raise ValueError("memory services must be unique")
        return self


class MemoryCreate(MemoryContent):
    """Confirm a resolution grounded in a completed, persisted source run."""

    source_run_id: UUID


class MemoryRevision(MemoryCreate):
    """Reconfirm corrected content while detecting concurrent edits."""

    expected_revision: int = Field(ge=1)


class MemoryRecord(MemoryCreate):
    """Stored confirmation with workflow-owned provenance and revision."""

    id: UUID = Field(default_factory=uuid4)
    revision: int = Field(default=1, ge=1)
    status: Literal["confirmed", "revoked"] = "confirmed"
    source_incident_id: UUID
    source_evidence_ids: list[UUID] = Field(min_length=1, max_length=20)
    omitted_source_evidence_ids: int = Field(default=0, ge=0)
    confirmed_at: AwareDatetime
    updated_at: AwareDatetime
    feedback_run_id: UUID | None = None
    feedback_quote: str | None = Field(default=None, max_length=400)


class MemorySnapshot(BaseModel):
    """Freeze selected record contents; edits affect only future retrieval."""

    model_config = ConfigDict(extra="forbid")
    scope: MemoryScope
    selected_at: AwareDatetime
    records: list[MemoryRecord] = Field(default_factory=list, max_length=10)

    @property
    def fingerprint(self) -> str:
        """Identify frozen contents, including revisions and reference time."""
        # Absent feedback provenance must preserve pre-feedback snapshot IDs.
        payload = json.dumps(self.model_dump(mode="json", exclude_none=True),
                             sort_keys=True,
                             ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def eligible_records(
        self, *, scope: MemoryScope, services: list[str],
        incident_id: UUID, at: datetime,
    ) -> list[MemoryRecord]:
        """Reject invalid scope, confirmation, time and same-incident data."""
        return [record for record in self.records if (
            record.status == "confirmed"
            and record.environment == scope.environment
            and record.deployment_version == scope.deployment_version
            and record.confirmed_at <= at < record.expires_at
            and record.source_incident_id != incident_id
            and set(services).intersection(record.services)
        )]


class FeedbackUpdate(BaseModel):
    """Agent-proposed learning; workflow verifies the quoted human source."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    kind: Literal["preference", "skill", "diagnosis"]
    action: Literal["upsert", "delete"] = "upsert"
    durability: Literal["persistent", "turn"] = "turn"
    key: Annotated[str, StringConstraints(
        pattern=r"^[a-z][a-z0-9_.-]{0,79}$",
    )] | None = None
    content: str = Field(default="", max_length=2000)
    quote: str = Field(min_length=1, max_length=400)
    rationale: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def validate_action(self) -> "FeedbackUpdate":
        """Require a stable guidance key and content for an upsert."""
        if self.kind != "diagnosis" and self.key is None:
            raise ValueError("guidance requires a stable key")
        if self.action == "upsert" and not self.content:
            raise ValueError("upsert requires content")
        return self


class GuidanceRecord(BaseModel):
    """Local user's reusable preference or skill, with feedback provenance."""

    model_config = ConfigDict(extra="forbid")
    id: UUID = Field(default_factory=uuid4)
    key: str
    kind: Literal["preference", "skill"]
    instruction: str = Field(min_length=1, max_length=2000)
    revision: int = Field(default=1, ge=1)
    status: Literal["active", "revoked"] = "active"
    origin: Literal["feedback", "manual"] = "feedback"
    source_run_id: UUID | None
    source_quote: str = Field(min_length=1, max_length=400)
    rationale: str = Field(min_length=1, max_length=300)
    updated_at: AwareDatetime


class GuidanceSnapshot(BaseModel):
    """Freeze local-user guidance independently of deployment identity."""

    profile: Literal["local-user"] = "local-user"
    selected_at: AwareDatetime
    records: list[GuidanceRecord] = Field(default_factory=list, max_length=20)


class GuidanceRevision(BaseModel):
    """Allow a human to correct an inferred instruction with a revision."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    instruction: str = Field(min_length=1, max_length=2000)
    expected_revision: int = Field(ge=1)


class FeedbackResult(BaseModel):
    """Idempotent write outcome; diagnostics expose metadata, not feedback."""

    applied: int = 0
    skipped: int = 0
    rejected: list[str] = Field(default_factory=list)
    guidance_ids: list[UUID] = Field(default_factory=list)
    memory_ids: list[UUID] = Field(default_factory=list)
    updates: list[FeedbackUpdate] = Field(default_factory=list, max_length=3)
