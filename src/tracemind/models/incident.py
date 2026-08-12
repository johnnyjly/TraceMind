"""Incoming incident descriptions and severity classifications."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class IncidentSeverity(StrEnum):
    """Reported incident severity, including an unknown default."""

    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Incident(BaseModel):
    """Incident description and affected-service hints for an
    investigation.
    """

    id: UUID = Field(default_factory=uuid4)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    severity: IncidentSeverity = IncidentSeverity.UNKNOWN
    affected_services: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
