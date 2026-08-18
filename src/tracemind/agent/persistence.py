"""SQLite run records, durable evidence snapshots, and worker leases."""

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

from tracemind.agent.service import DiagnosisRun
from tracemind.models import Evidence, Incident
from tracemind.models.run import (
    RunEvent, RunLimits, RunPurpose, RunState, RunStatus, StopReason, ToolName,
)


SCHEMA_VERSION = 2
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
            if version not in (0, 1, SCHEMA_VERSION):
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
