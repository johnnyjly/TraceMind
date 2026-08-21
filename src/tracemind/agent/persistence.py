"""SQLite run records, durable evidence snapshots, and worker leases."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterator
from uuid import UUID, uuid4

from pydantic import ValidationError

from tracemind.agent.service import DiagnosisRun
from tracemind.models import Evidence, Incident
from tracemind.models.memory import (
    FeedbackResult, FeedbackUpdate, GuidanceRecord, GuidanceRevision,
    GuidanceSnapshot,
    MemoryCreate, MemoryPolicy, MemoryRecord, MemorySnapshot,
)
from tracemind.models.run import (
    RunEvent, RunLimits, RunPurpose, RunState, RunStatus, StopReason, ToolName,
)


SCHEMA_VERSION = 4
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


class RunRepository:
    """Keep local run snapshots and atomically fence one worker per run.

    The checkpoint database is separate because LangGraph owns its tables and
    transaction boundaries. SQLite is intended for one local worker process.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                                 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2, 3, SCHEMA_VERSION):
                raise RuntimeError(
                    f"unsupported run database schema version: {version}"
                )
            if version == 0:
                connection.execute("""
                    CREATE TABLE runs (
                        id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        state_json TEXT NOT NULL,
                        evidence_json TEXT NOT NULL,
                        result_json TEXT,
                        request_hash TEXT NOT NULL,
                        idempotency_key TEXT UNIQUE,
                        cancel_requested INTEGER NOT NULL DEFAULT 0,
                        lease_owner TEXT,
                        lease_until REAL,
                        attempt INTEGER NOT NULL DEFAULT 0,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                """)
            if version < 2:
                connection.execute("""
                    CREATE TABLE investigations (
                        id TEXT PRIMARY KEY,
                        incident_json TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                """)
                connection.execute(
                    "ALTER TABLE runs ADD COLUMN investigation_id TEXT"
                )
                connection.execute("""
                    CREATE INDEX runs_investigation ON runs(investigation_id)
                """)
                connection.execute("PRAGMA user_version=2")
            if version < 3:
                connection.execute("""
                    CREATE TABLE incident_memories (
                        id TEXT PRIMARY KEY,
                        revision INTEGER NOT NULL,
                        environment TEXT NOT NULL,
                        deployment_version TEXT NOT NULL,
                        source_incident_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        confirmed_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        record_json TEXT NOT NULL
                    )
                """)
                connection.execute("""
                    CREATE INDEX memories_scope ON incident_memories
                    (environment,deployment_version,status,expires_at)
                """)
                connection.execute("PRAGMA user_version=3")
            if version < 4:
                connection.execute("""
                    CREATE TABLE agent_guidance (
                        id TEXT PRIMARY KEY,
                        key TEXT NOT NULL UNIQUE,
                        record_json TEXT NOT NULL
                    )
                """)
                connection.execute("""
                    CREATE TABLE memory_feedback (
                        run_id TEXT PRIMARY KEY,
                        result_json TEXT NOT NULL
                    )
                """)
                connection.execute("PRAGMA user_version=4")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _evidence_json(evidence: list[Evidence]) -> str:
        return json.dumps([item.model_dump(mode="json") for item in evidence])

    @staticmethod
    def _decode(row: sqlite3.Row) -> StoredRun:
        evidence = [Evidence.model_validate(item) for item in
                    json.loads(row["evidence_json"])]
        result = (DiagnosisRun.model_validate_json(row["result_json"])
                  if row["result_json"] else None)
        return StoredRun(
            state=RunState.model_validate_json(row["state_json"]),
            evidence=evidence,
            result=result,
            cancel_requested=bool(row["cancel_requested"]),
            attempt=row["attempt"],
            lease_until=row["lease_until"],
        )

    def create(
        self, state: RunState, *, request_hash: str,
        idempotency_key: str | None = None,
        new_investigation: bool = False,
    ) -> StoredRun:
        """Queue one run, reusing matching idempotency keys atomically."""
        if state.status is not RunStatus.QUEUED:
            raise ValueError("new persistent runs must be queued")
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if idempotency_key is not None:
                row = connection.execute(
                    "SELECT * FROM runs WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                if row is not None:
                    if row["request_hash"] != request_hash:
                        raise RunConflict(
                            "idempotency key belongs to a different request"
                        )
                    return self._decode(row)
            if new_investigation:
                if state.investigation_id is None:
                    raise ValueError("an investigation ID is required")
                connection.execute(
                    "INSERT INTO investigations VALUES (?,?,?)",
                    (str(state.investigation_id),
                     state.incident.model_dump_json(), now),
                )
            elif state.investigation_id is not None:
                raise ValueError("use follow_up to extend an investigation")
            self._insert_run(connection, state, [], request_hash,
                             idempotency_key, now)
        return self.get(state.id)

    @staticmethod
    def _append_event(state: RunState, kind: str, data: dict) -> None:
        """Record workflow transitions even when no harness is running."""
        state.events.append(RunEvent(
            run_id=state.id, sequence=len(state.events) + 1,
            kind=kind, span_id=state.id, data=data,
        ))

    def _insert_run(
        self, connection: sqlite3.Connection, state: RunState,
        evidence: list[Evidence], request_hash: str,
        idempotency_key: str | None, now: float,
    ) -> None:
        """Insert a queued round and its retained observations atomically."""
        state.evidence_ids = [item.id for item in evidence]
        self._append_event(state, "run.queued", {})
        connection.execute(
            """INSERT INTO runs
               (id,status,state_json,evidence_json,request_hash,
                idempotency_key,created_at,updated_at,investigation_id)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (str(state.id), state.status.value, state.model_dump_json(),
             self._evidence_json(evidence), request_hash, idempotency_key,
             now, now, str(state.investigation_id)
             if state.investigation_id else None),
        )

    def get_investigation(
        self, investigation_id: UUID, *, offset: int = 0, limit: int = 20,
    ) -> StoredInvestigation:
        """Read incident metadata and ordered round summaries."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM investigations WHERE id=?",
                (str(investigation_id),),
            ).fetchone()
            if row is None:
                raise RunNotFound(str(investigation_id))
            runs = connection.execute(
                """SELECT state_json FROM runs WHERE investigation_id=?
                   ORDER BY created_at,rowid LIMIT ? OFFSET ?""",
                (str(investigation_id), limit + 1, offset),
            ).fetchall()
        return StoredInvestigation(
            id=investigation_id,
            incident=Incident.model_validate_json(row["incident_json"]),
            created_at=datetime.fromtimestamp(row["created_at"], UTC),
            runs=[RunState.model_validate_json(r[0]) for r in runs[:limit]],
            next_offset=offset + limit if len(runs) > limit else None,
        )

    def follow_up(
        self, investigation_id: UUID, *, message: str, purpose: RunPurpose,
        limits: RunLimits, allowed_tools: tuple[ToolName, ...],
        start: datetime | None = None, end: datetime | None = None,
        request_hash: str, idempotency_key: str | None = None,
        memory_policy: MemoryPolicy | None = None,
    ) -> StoredRun:
        """Append one round after the previous round commits its outcome.

        Prior observations are copied by value within the new window and
        capacity, with cited observations selected first. User messages and
        previous diagnoses are hypotheses, not observations.
        """
        if purpose is RunPurpose.AUTOMATIC:
            raise ValueError("follow-up requires investigate or explain")
        if (start is None) != (end is None):
            raise ValueError("provide both time bounds or neither")
        if not message.strip() or len(message) > 4000:
            raise ValueError("follow-up requires 1-4000 nonblank characters")
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if idempotency_key is not None:
                repeated = connection.execute(
                    "SELECT * FROM runs WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                if repeated is not None:
                    if (repeated["request_hash"] != request_hash or
                            repeated["investigation_id"] !=
                            str(investigation_id)):
                        raise RunConflict("idempotency key request differs")
                    return self._decode(repeated)
            rows = connection.execute(
                """SELECT * FROM runs WHERE investigation_id=?
                   ORDER BY created_at DESC,rowid DESC LIMIT 3""",
                (str(investigation_id),),
            ).fetchall()
            if not rows:
                raise RunNotFound(str(investigation_id))
            parent = self._decode(rows[0])
            if RunStatus(rows[0]["status"]) not in TERMINAL:
                raise RunConflict("finish or cancel the active round first")
            start = start or parent.state.investigation_start
            end = end or parent.state.investigation_end
            if (start.utcoffset() is None or end.utcoffset() is None
                    or end <= start):
                raise ValueError("invalid follow-up investigation window")
            if purpose is RunPurpose.EXPLAIN and (
                start != parent.state.investigation_start or
                end != parent.state.investigation_end
            ):
                raise RunConflict("explanation must use the previous window")
            prior = parent.state.diagnosis or parent.state.prior_diagnosis
            candidates = [item for item in parent.evidence
                          if item.observed_at.utcoffset() is not None
                          and start <= item.observed_at <= end]
            cited = set(prior.evidence_ids) if prior else set()
            candidates.sort(key=lambda item: item.id not in cited)
            evidence = []
            size = 0
            for item in candidates:
                item_size = len(item.model_dump_json().encode())
                if size + item_size <= limits.observation_bytes:
                    evidence.append(item)
                    size += item_size
            history = [RunState.model_validate_json(row["state_json"])
                       for row in reversed(rows)]
            state = RunState(
                incident=parent.state.incident,
                investigation_id=investigation_id,
                parent_run_id=parent.state.id, purpose=purpose,
                prompt_version="single-agent-followup-v2",
                user_message=message,
                previous_messages=[s.user_message for s in history
                                   if s.user_message],
                prior_diagnosis=prior,
                omitted_prior_evidence=len(parent.evidence) - len(evidence),
                memory_policy=(memory_policy if purpose is
                               RunPurpose.INVESTIGATE else None),
                learning_policy=memory_policy,
                investigation_start=start, investigation_end=end,
                limits=limits.model_copy(deep=True),
                allowed_tools=(tuple(t for t in allowed_tools
                                     if t == "read_evidence")
                               if purpose is RunPurpose.EXPLAIN
                               else allowed_tools),
                status=RunStatus.QUEUED,
            )
            self._insert_run(connection, state, evidence, request_hash,
                             idempotency_key, now)
        return self.get(state.id)

    def event_snapshot(self, run_id: UUID) -> EventSnapshot:
        """Project persisted events without loading large evidence payloads."""
        with self._connect() as connection:
            row = connection.execute(
                """SELECT status,state_json,result_json IS NOT NULL AS result
                   FROM runs WHERE id=?""", (str(run_id),),
            ).fetchone()
        if row is None:
            raise RunNotFound(str(run_id))
        events = json.loads(row["state_json"])["events"]
        return EventSnapshot(
            RunStatus(row["status"]),
            [RunEvent.model_validate(event) for event in events],
            bool(row["result"]),
        )

    def get(self, run_id: UUID) -> StoredRun:
        """Read a consistent run snapshot by ID."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE id=?", (str(run_id),)
            ).fetchone()
        if row is None:
            raise RunNotFound(str(run_id))
        return self._decode(row)

    def claim_next(self, *, lease_seconds: float = 30) -> ClaimedRun | None:
        """Claim a queued run or a worker's expired lease."""
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            while True:
                row = connection.execute(
                    """SELECT * FROM runs WHERE status='queued' OR
                        (status='running' AND lease_until<?)
                       ORDER BY created_at LIMIT 1""", (now,),
                ).fetchone()
                if row is None:
                    return None
                stored = self._decode(row)
                state = stored.state
                if stored.cancel_requested and state.status not in TERMINAL:
                    state.status = RunStatus.CANCELLED
                    state.stop_reason = StopReason.CANCELLED
                if state.status in TERMINAL:
                    self._ensure_terminal_event(state)
                    diagnosis = state.diagnosis
                    result = (
                        DiagnosisRun(
                            diagnosis=diagnosis, evidence=stored.evidence,
                            tool_calls=state.tool_calls, state=state,
                        ).model_dump_json() if diagnosis else None
                    )
                    connection.execute(
                        """UPDATE runs SET status=?,state_json=?,result_json=?,
                           lease_owner=NULL,lease_until=NULL,updated_at=?
                           WHERE id=?""",
                        (state.status.value, state.model_dump_json(),
                         result, now, row["id"]),
                    )
                    continue
                resuming = row["attempt"] > 0
                owner = str(uuid4())
                state.status = RunStatus.RUNNING
                if not resuming:
                    state.started_at = datetime.now(UTC)
                    state.deadline_at = state.started_at + timedelta(
                        seconds=state.limits.duration_seconds
                    )
                connection.execute(
                    """UPDATE runs SET status='running',state_json=?,
                       lease_owner=?,lease_until=?,attempt=attempt+1,
                       updated_at=? WHERE id=?""",
                    (state.model_dump_json(), owner, now + lease_seconds,
                     now, row["id"]),
                )
                return ClaimedRun(
                    StoredRun(state, stored.evidence, None, False,
                              stored.attempt + 1, now + lease_seconds),
                    owner, resuming,
                )

    def save_progress(
        self, run_id: UUID, owner: str, state: RunState,
        evidence: list[Evidence],
    ) -> None:
        """Persist a snapshot only while this worker owns the lease."""
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE runs SET state_json=?,evidence_json=?,updated_at=?
                   WHERE id=? AND lease_owner=? AND status='running'""",
                (state.model_dump_json(), self._evidence_json(evidence),
                 time.time(), str(run_id), owner),
            )
            if cursor.rowcount != 1:
                raise RunConflict("worker lease was lost")

    def heartbeat(
        self, run_id: UUID, owner: str, *, lease_seconds: float = 30,
    ) -> bool:
        """Extend the lease and report a persisted cancellation request."""
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE runs SET lease_until=?,updated_at=?
                   WHERE id=? AND lease_owner=? AND status='running'""",
                (time.time() + lease_seconds, time.time(),
                 str(run_id), owner),
            )
            if cursor.rowcount != 1:
                raise RunConflict("worker lease was lost")
            row = connection.execute(
                "SELECT cancel_requested FROM runs WHERE id=?",
                (str(run_id),),
            ).fetchone()
            return bool(row[0])

    def should_stop(self, run_id: UUID, owner: str) -> bool:
        """Prevent new work after cancellation or loss of ownership."""
        with self._connect() as connection:
            row = connection.execute(
                """SELECT cancel_requested,lease_owner FROM runs
                   WHERE id=? AND status='running'""", (str(run_id),)
            ).fetchone()
        return (
            row is None or row["cancel_requested"]
            or row["lease_owner"] != owner
        )

    def finish(
        self, run_id: UUID, owner: str, state: RunState,
        evidence: list[Evidence],
    ) -> None:
        """Commit the terminal snapshot and result under the worker lease."""
        if state.status not in TERMINAL:
            raise ValueError("cannot finish a nonterminal run")
        self._ensure_terminal_event(state)
        result = (
            DiagnosisRun(
                diagnosis=state.diagnosis, evidence=evidence,
                tool_calls=state.tool_calls, state=state,
            ).model_dump_json() if state.diagnosis is not None else None
        )
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE runs SET status=?,state_json=?,evidence_json=?,
                   result_json=?,lease_owner=NULL,lease_until=NULL,
                   updated_at=? WHERE id=? AND lease_owner=?""",
                (state.status.value, state.model_dump_json(),
                 self._evidence_json(evidence), result, time.time(),
                 str(run_id), owner),
            )
            if cursor.rowcount != 1:
                raise RunConflict("worker lease was lost")

    def _ensure_terminal_event(self, state: RunState) -> None:
        """Give early failure/cancellation paths a durable completion event."""
        if not any(e.kind == "run.finished" for e in state.events):
            state.finished_at = state.finished_at or datetime.now(UTC)
            self._append_event(state, "run.finished", {
                "status": state.status.value,
                "reason": state.stop_reason.value if state.stop_reason
                else None,
            })

    def cancel(self, run_id: UUID) -> StoredRun:
        """Cancel queued/stale runs immediately or signal the active worker."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM runs WHERE id=?", (str(run_id),)
            ).fetchone()
            if row is None:
                raise RunNotFound(str(run_id))
            if row["status"] not in {item.value for item in TERMINAL}:
                state = RunState.model_validate_json(row["state_json"])
                if state.status in TERMINAL:
                    # Finalization may have saved its snapshot just before
                    # the worker commits the result. Keep that outcome.
                    self._ensure_terminal_event(state)
                    result = (DiagnosisRun(
                        diagnosis=state.diagnosis,
                        evidence=self._decode(row).evidence,
                        tool_calls=state.tool_calls, state=state,
                    ).model_dump_json() if state.diagnosis else None)
                    connection.execute(
                        """UPDATE runs SET status=?,state_json=?,result_json=?,
                           lease_owner=NULL,lease_until=NULL,updated_at=?
                           WHERE id=?""",
                        (state.status.value, state.model_dump_json(), result,
                         time.time(), str(run_id)),
                    )
                elif row["status"] == "queued" or (
                    row["lease_until"] is not None
                    and row["lease_until"] < time.time()
                ):
                    state.status = RunStatus.CANCELLED
                    state.stop_reason = StopReason.CANCELLED
                    state.finished_at = datetime.now(UTC)
                    self._ensure_terminal_event(state)
                    connection.execute(
                        """UPDATE runs SET status='cancelled',state_json=?,
                           cancel_requested=1,lease_owner=NULL,
                           lease_until=NULL,updated_at=? WHERE id=?""",
                        (state.model_dump_json(), time.time(), str(run_id)),
                    )
                else:
                    connection.execute(
                        """UPDATE runs SET cancel_requested=1,updated_at=?
                           WHERE id=?""",
                        (time.time(), str(run_id)),
                    )
        return self.get(run_id)

    def resume(self, run_id: UUID) -> StoredRun:
        """Requeue an interrupted run only after its prior lease expires."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM runs WHERE id=?", (str(run_id),)
            ).fetchone()
            if row is None:
                raise RunNotFound(str(run_id))
            if row["status"] != "running" or row["cancel_requested"]:
                raise RunConflict("run is not interrupted and resumable")
            if (row["lease_until"] is not None
                    and row["lease_until"] >= time.time()):
                raise RunConflict("run is still owned by an active worker")
            state = RunState.model_validate_json(row["state_json"])
            if state.status in TERMINAL:
                raise RunConflict("run has a terminal outcome")
            state.status = RunStatus.QUEUED
            connection.execute(
                """UPDATE runs SET status='queued',state_json=?,
                   lease_owner=NULL,lease_until=NULL,updated_at=?
                   WHERE id=?""",
                (state.model_dump_json(), time.time(), str(run_id)),
            )
        return self.get(run_id)


class MemoryNotFound(KeyError):
    """The historical experience has been deleted or never existed."""


class MemoryRepository:
    """Manage human confirmations using the existing local run database.

    The agent has no write access. Revisions use optimistic concurrency;
    previously captured run snapshots retain the exact historical content.
    """

    def __init__(self, runs: RunRepository) -> None:
        """Reuse SQLite transactions and source-run provenance checks."""
        self.runs = runs

    def get(self, memory_id: UUID) -> MemoryRecord:
        """Read current content, including revoked or expired records."""
        with self.runs._connect() as connection:
            row = connection.execute(
                "SELECT record_json FROM incident_memories WHERE id=?",
                (str(memory_id),),
            ).fetchone()
        if row is None:
            raise MemoryNotFound(str(memory_id))
        return MemoryRecord.model_validate_json(row[0])

    def list(self, *, offset: int = 0, limit: int = 20) -> list[MemoryRecord]:
        """Page current records for manual review, without hiding expiry."""
        with self.runs._connect() as connection:
            rows = connection.execute(
                """SELECT record_json FROM incident_memories
                   ORDER BY updated_at DESC,id LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
        return [MemoryRecord.model_validate_json(row[0]) for row in rows]

    def confirm(
        self, content: MemoryCreate, *, memory_id: UUID | None = None,
        expected_revision: int | None = None,
    ) -> MemoryRecord:
        """Create or reconfirm content after explicit human review.

        A successful model run alone never calls this method. The resolution
        and confirmation note are human assertions, not machine verification.
        """
        now = datetime.now(UTC)
        if content.expires_at <= now:
            raise RunConflict("memory expiry must be in the future")
        with self.runs._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            revision = 1
            if memory_id is not None:
                current = connection.execute(
                    "SELECT revision FROM incident_memories WHERE id=?",
                    (str(memory_id),),
                ).fetchone()
                if current is None:
                    raise MemoryNotFound(str(memory_id))
                if current[0] != expected_revision:
                    raise RunConflict("memory revision changed; reload it")
                revision = current[0] + 1
            source = connection.execute(
                "SELECT * FROM runs WHERE id=?", (str(content.source_run_id),),
            ).fetchone()
            if source is None:
                raise RunNotFound(str(content.source_run_id))
            run = self.runs._decode(source)
            if (run.state.status is not RunStatus.SUCCEEDED or
                    run.result is None or
                    not run.result.diagnosis.evidence_ids):
                raise RunConflict(
                    "memory source must be a completed diagnosis")
            known = {e.id: e for e in run.evidence}
            citations = list(dict.fromkeys(run.result.diagnosis.evidence_ids))
            if any(
                i not in known or
                known[i].observed_at.utcoffset() is None or not (
                    run.state.investigation_start <= known[i].observed_at
                    <= run.state.investigation_end
                ) for i in citations
            ):
                raise RunConflict(
                    "memory source has invalid evidence citations")
            services = set(run.state.incident.affected_services) | {
                e.service for e in run.evidence if e.service
            }
            if not set(content.services) <= services:
                raise RunConflict(
                    "memory services must occur in the source run")
            source_policy = run.state.memory_policy
            if source_policy is not None and (
                content.environment != source_policy.scope.environment or
                content.deployment_version !=
                    source_policy.scope.deployment_version
            ):
                raise RunConflict("memory scope differs from the source run")
            data = content.model_dump(exclude={"expected_revision"})
            record = MemoryRecord(
                **data, id=memory_id or uuid4(), revision=revision,
                source_incident_id=run.state.incident.id,
                source_evidence_ids=citations[:20],
                omitted_source_evidence_ids=max(0, len(citations) - 20),
                confirmed_at=now, updated_at=now,
            )
            connection.execute(
                """INSERT INTO incident_memories
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                   revision=excluded.revision,environment=excluded.environment,
                   deployment_version=excluded.deployment_version,
                   source_incident_id=excluded.source_incident_id,
                   status=excluded.status,confirmed_at=excluded.confirmed_at,
                   expires_at=excluded.expires_at,updated_at=excluded.updated_at,
                   record_json=excluded.record_json""",
                (str(record.id), revision, record.environment,
                 record.deployment_version, str(record.source_incident_id),
                 record.status, now.timestamp(), record.expires_at.timestamp(),
                 now.timestamp(), record.model_dump_json()),
            )
        return record

    def revoke(
        self, memory_id: UUID, *, expected_revision: int,
    ) -> MemoryRecord:
        """Stop future retrieval while retaining the record for review."""
        with self.runs._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT record_json FROM incident_memories WHERE id=?",
                (str(memory_id),),
            ).fetchone()
            if row is None:
                raise MemoryNotFound(str(memory_id))
            current = MemoryRecord.model_validate_json(row[0])
            if current.revision != expected_revision:
                raise RunConflict("memory revision changed; reload it")
            current.status = "revoked"
            current.revision += 1
            current.updated_at = datetime.now(UTC)
            connection.execute(
                """UPDATE incident_memories SET revision=?,status='revoked',
                   updated_at=?,record_json=? WHERE id=?""",
                (current.revision, current.updated_at.timestamp(),
                 current.model_dump_json(), str(memory_id)),
            )
        return current

    def delete(self, memory_id: UUID, *, expected_revision: int) -> None:
        """Delete current content; prior run snapshots retain their audit."""
        with self.runs._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM incident_memories WHERE id=? AND revision=?",
                (str(memory_id), expected_revision),
            )
            if cursor.rowcount != 1:
                raise RunConflict("memory missing or revision changed")

    def snapshot(
        self, policy: MemoryPolicy, incident: Incident, at: datetime,
    ) -> MemorySnapshot:
        """Select current, same-deployment experience from other incidents."""
        services = incident.affected_services
        records = []
        if services:
            placeholders = ",".join("?" for _ in services)
            with self.runs._connect() as connection:
                rows = connection.execute(
                    f"""SELECT record_json FROM incident_memories
                        WHERE environment=? AND deployment_version=?
                        AND status='confirmed' AND expires_at>?
                        AND confirmed_at<=? AND source_incident_id!=?
                        AND EXISTS (SELECT 1 FROM json_each(record_json,
                            '$.services') WHERE value IN ({placeholders}))
                        ORDER BY updated_at DESC,id LIMIT ?""",
                    (policy.scope.environment, policy.scope.deployment_version,
                     at.timestamp(), at.timestamp(), str(incident.id),
                     *services, policy.max_records),
                ).fetchall()
            records = [MemoryRecord.model_validate_json(row[0])
                       for row in rows]
        return MemorySnapshot(scope=policy.scope, selected_at=at,
                              records=records)

    def guidance_snapshot(self) -> GuidanceSnapshot:
        """Retrieve bounded local-user guidance, independent of deployment."""
        with self.runs._connect() as connection:
            rows = connection.execute(
                """SELECT record_json FROM agent_guidance
                   WHERE json_extract(record_json,'$.status')='active'
                   ORDER BY json_extract(record_json,'$.updated_at') DESC,key
                   LIMIT 20""",
            ).fetchall()
        return GuidanceSnapshot(selected_at=datetime.now(UTC), records=[
            GuidanceRecord.model_validate_json(row[0]) for row in rows
        ])

    def list_guidance(
        self, *, offset: int = 0, limit: int = 20,
    ) -> list[GuidanceRecord]:
        """Review all active or revoked local-user instructions."""
        with self.runs._connect() as connection:
            rows = connection.execute(
                "SELECT record_json FROM agent_guidance ORDER BY key "
                "LIMIT ? OFFSET ?", (limit, offset),
            ).fetchall()
        return [GuidanceRecord.model_validate_json(row[0]) for row in rows]

    def revise_guidance(
        self, guidance_id: UUID, revision: GuidanceRevision,
    ) -> GuidanceRecord:
        """Correct a learned instruction; preserve old run snapshots."""
        with self.runs._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT record_json FROM agent_guidance WHERE id=?",
                (str(guidance_id),),
            ).fetchone()
            if row is None:
                raise MemoryNotFound(str(guidance_id))
            record = GuidanceRecord.model_validate_json(row[0])
            if record.revision != revision.expected_revision:
                raise RunConflict("guidance revision changed; reload it")
            record.instruction = revision.instruction
            record.status = "active"
            record.revision += 1
            record.origin = "manual"
            record.source_run_id = None
            record.source_quote = revision.instruction[:400]
            record.rationale = "Corrected through the management API."
            record.updated_at = datetime.now(UTC)
            connection.execute(
                "UPDATE agent_guidance SET record_json=? WHERE id=?",
                (record.model_dump_json(), str(guidance_id)),
            )
        return record

    def delete_guidance(
        self, guidance_id: UUID, *, expected_revision: int,
    ) -> None:
        """Remove an instruction with optimistic concurrency protection."""
        with self.runs._connect() as connection:
            cursor = connection.execute(
                """DELETE FROM agent_guidance WHERE id=?
                   AND json_extract(record_json,'$.revision')=?""",
                (str(guidance_id), expected_revision),
            )
            if cursor.rowcount != 1:
                raise RunConflict("guidance missing or revision changed")

    def apply_feedback(self, state: RunState, *, owner: str) -> FeedbackResult:
        """Validate agent proposals and atomically learn once per leased run.

        Proposals must quote the current human message. The durable receipt
        makes replay safe even if a crash precedes the workflow progress event.
        Conflicts are reported; SQLite failures propagate.
        """
        with self.runs._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            lease = connection.execute(
                """SELECT lease_owner,status,cancel_requested FROM runs
                   WHERE id=?""", (str(state.id),),
            ).fetchone()
            if (lease is None or lease["lease_owner"] != owner or
                    lease["status"] != "running" or lease["cancel_requested"]):
                raise RunConflict("feedback run lease lost or cancelled")
            receipt = connection.execute(
                "SELECT result_json FROM memory_feedback WHERE run_id=?",
                (str(state.id),),
            ).fetchone()
            if receipt is not None:
                return FeedbackResult.model_validate_json(receipt[0])
            result = FeedbackResult()
            if state.learning_policy is None or not state.user_message:
                return result
            result.updates = [u.model_copy(deep=True)
                              for u in state.feedback_updates]
            for update in state.feedback_updates:
                if update.durability == "turn":
                    result.skipped += 1
                    continue
                if update.quote not in state.user_message:
                    result.rejected.append("quote_not_in_current_feedback")
                    continue
                try:
                    if update.kind == "diagnosis":
                        record_id = self._learn_diagnosis(connection,
                                                          state, update)
                        if record_id is not None:
                            result.memory_ids.append(record_id)
                    else:
                        record_id = self._learn_guidance(connection,
                                                         state, update)
                        if record_id is not None:
                            result.guidance_ids.append(record_id)
                except RunConflict as exc:
                    result.rejected.append(str(exc))
                except ValidationError:
                    result.rejected.append("memory_record_validation")
                else:
                    if record_id is None:
                        result.skipped += 1
                    else:
                        result.applied += 1
            connection.execute("INSERT INTO memory_feedback VALUES (?,?)",
                               (str(state.id), result.model_dump_json()))
        return result

    def _learn_guidance(
        self, connection: sqlite3.Connection, state: RunState,
        update: FeedbackUpdate,
    ) -> UUID | None:
        """Reject stale learning using the captured key revision."""
        row = connection.execute(
            "SELECT record_json FROM agent_guidance WHERE key=?",
            (update.key,),
        ).fetchone()
        current = GuidanceRecord.model_validate_json(row[0]) if row else None
        if state.guidance_snapshot is None:
            raise RunConflict("guidance_snapshot_missing")
        captured = next((r for r in state.guidance_snapshot.records
                         if r.key == update.key), None)
        recreating = (current is not None and current.status == "revoked"
                      and captured is None and
                      current.updated_at <=
                      state.guidance_snapshot.selected_at)
        if (current is not None and not recreating and (
            captured is None or current.id != captured.id or
            current.revision != captured.revision
        )) or (current is None and captured is not None):
            raise RunConflict("guidance_revision_conflict")
        if update.action == "delete":
            if current is not None:
                current.status = "revoked"
                current.origin = "feedback"
                current.revision += 1
                current.source_run_id = state.id
                current.source_quote = update.quote
                current.rationale = update.rationale
                current.updated_at = datetime.now(UTC)
                connection.execute(
                    "UPDATE agent_guidance SET record_json=? WHERE id=?",
                    (current.model_dump_json(), str(current.id)),
                )
            return current.id if current else None
        record = GuidanceRecord(
            id=current.id if current else uuid4(), key=update.key,
            kind=update.kind, instruction=update.content,
            revision=current.revision + 1 if current else 1,
            source_run_id=state.id, source_quote=update.quote,
            rationale=update.rationale, updated_at=datetime.now(UTC),
        )
        connection.execute(
            """INSERT INTO agent_guidance VALUES (?,?,?) ON CONFLICT(key)
               DO UPDATE SET record_json=excluded.record_json""",
            (str(record.id), record.key, record.model_dump_json()),
        )
        return record.id

    def _learn_diagnosis(
        self, connection: sqlite3.Connection, state: RunState,
        update: FeedbackUpdate,
    ) -> UUID | None:
        """Associate human recovery/correction feedback with its parent run."""
        row = connection.execute("SELECT * FROM runs WHERE id=?",
                                 (str(state.parent_run_id),)).fetchone()
        if row is None:
            raise RunConflict("diagnosis_source_missing")
        source = self.runs._decode(row)
        if source.state.incident.id != state.incident.id:
            raise RunConflict("diagnosis_source_incident_mismatch")
        scope = state.learning_policy.scope
        source_scope = (source.state.learning_policy or
                        source.state.memory_policy)
        if source_scope is not None and source_scope.scope != scope:
            raise RunConflict("diagnosis_source_scope_mismatch")
        prior = connection.execute(
            """SELECT record_json FROM incident_memories
               WHERE source_incident_id=? AND environment=?
               AND deployment_version=? ORDER BY updated_at DESC LIMIT 1""",
            (str(state.incident.id), scope.environment,
             scope.deployment_version),
        ).fetchone()
        current = MemoryRecord.model_validate_json(prior[0]) if prior else None
        if current is not None and current.updated_at > state.started_at:
            raise RunConflict("diagnosis_revision_conflict")
        if update.action == "delete":
            if current is not None:
                current.status = "revoked"
                current.revision += 1
                current.updated_at = datetime.now(UTC)
                current.feedback_run_id = state.id
                current.feedback_quote = update.quote
                connection.execute(
                    """UPDATE incident_memories SET revision=?,status=?,
                       updated_at=?,record_json=? WHERE id=?""",
                    (current.revision, current.status,
                     current.updated_at.timestamp(), current.model_dump_json(),
                     str(current.id)),
                )
            return current.id if current else None
        anchors = [e for e in source.evidence
                   if e.observed_at.utcoffset() is not None and
                   source.state.investigation_start <= e.observed_at
                   <= source.state.investigation_end]
        if not anchors:
            raise RunConflict("diagnosis_source_has_no_observations")
        services = list(dict.fromkeys([
            *source.state.incident.affected_services,
            *(e.service for e in anchors if e.service),
        ]))[:20]
        if not services:
            raise RunConflict("diagnosis_source_has_no_services")
        now = datetime.now(UTC)
        record = MemoryRecord(
            id=current.id if current else uuid4(),
            revision=current.revision + 1 if current else 1,
            environment=scope.environment,
            deployment_version=scope.deployment_version, services=services,
            symptom=source.state.incident.title[:1000],
            resolution=update.content, confirmed_by="local-user-feedback",
            confirmation_note=f"{update.rationale}\nFeedback: {update.quote}",
            expires_at=now + timedelta(
                days=state.learning_policy.auto_expiry_days),
            source_run_id=source.state.id,
            source_incident_id=source.state.incident.id,
            source_evidence_ids=[e.id for e in anchors[:20]],
            omitted_source_evidence_ids=max(0, len(anchors) - 20),
            confirmed_at=now, updated_at=now,
            feedback_run_id=state.id, feedback_quote=update.quote,
        )
        connection.execute(
            """INSERT INTO incident_memories VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET revision=excluded.revision,
               status=excluded.status,confirmed_at=excluded.confirmed_at,
               expires_at=excluded.expires_at,updated_at=excluded.updated_at,
               record_json=excluded.record_json""",
            (str(record.id), record.revision, record.environment,
             record.deployment_version, str(record.source_incident_id),
             record.status, now.timestamp(), record.expires_at.timestamp(),
             now.timestamp(), record.model_dump_json()),
        )
        return record.id
