"""Normalized telemetry observations with provenance and stable IDs."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, JsonValue


class EvidenceSource(StrEnum):
    """Telemetry source category of a normalized observation."""

    METRIC = "metric"
    TRACE = "trace"
    LOG = "log"


class Evidence(BaseModel):
    """Observation with its source, time, query provenance, and backend
    details.
    """

    id: UUID = Field(default_factory=uuid4)
    source: EvidenceSource
    summary: str = Field(min_length=1)
    observed_at: datetime
    service: str | None = None
    query: str | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)
