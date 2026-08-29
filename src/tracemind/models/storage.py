"""Stored run snapshots and lifecycle conflicts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from tracemind.models import Evidence, Incident
from tracemind.models.result import DiagnosisRun
from tracemind.models.run import RunEvent, RunState, RunStatus


TERMINAL = {
    RunStatus.SUCCEEDED, RunStatus.INCONCLUSIVE,
    RunStatus.FAILED, RunStatus.CANCELLED,
}


class RunNotFound(KeyError):
    """A requested run does not exist."""


class RunConflict(ValueError):
    """A run operation conflicts with its persisted lifecycle."""


@dataclass(frozen=True)
class StoredRun:
    """Queryable snapshot and optional terminal diagnosis."""

    state: RunState
    evidence: list[Evidence]
    result: DiagnosisRun | None
    cancel_requested: bool
    attempt: int
    lease_until: float | None


@dataclass(frozen=True)
class ClaimedRun:
    """Exclusive worker claim and the state to resume."""

    run: StoredRun
    owner: str
    resuming: bool


@dataclass(frozen=True)
class StoredInvestigation:
    """One incident with a paginated history of immutable run outcomes."""

    id: UUID
    incident: Incident
    created_at: datetime
    runs: list[RunState]
    next_offset: int | None


@dataclass(frozen=True)
class EventSnapshot:
    """Read event history without decoding stored observations or results."""

    status: RunStatus
    events: list[RunEvent]
    has_result: bool
