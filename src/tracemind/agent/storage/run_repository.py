"""SQLite run records, durable evidence snapshots, and worker leases."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sqlite3
import time
from uuid import UUID, uuid4

from tracemind.agent.storage.database import RunDatabase
from tracemind.agent.workflow.followup import plan_follow_up
from tracemind.agent.storage.run_codec import decode_run
from tracemind.models import Evidence, Incident
from tracemind.models.memory import MemoryPolicy
from tracemind.models.result import DiagnosisRun
from tracemind.models.run import (
    RunEvent,
    RunLimits,
    RunPurpose,
    RunState,
    RunStatus,
    StopReason,
    ToolName,
)
from tracemind.models.storage import (
    ClaimedRun,
    EventSnapshot,
    RunConflict,
    RunNotFound,
    StoredInvestigation,
    StoredRun,
    TERMINAL,
)


class RunRepository:
    """Keep local run snapshots and atomically fence one worker per run.

    The checkpoint database is separate because LangGraph owns its tables and
    transaction boundaries. SQLite is intended for one local worker process.
    """

    def __init__(self, path: Path | RunDatabase) -> None:
        self.database = (path if isinstance(path, RunDatabase)
                         else RunDatabase(path))
        self.path = self.database.path

    @staticmethod
    def _evidence_json(evidence: list[Evidence]) -> str:
        return json.dumps([item.model_dump(mode="json") for item in evidence])

    def create(
        self, state: RunState, *, request_hash: str,
        idempotency_key: str | None = None,
        new_investigation: bool = False,
    ) -> StoredRun:
        """Queue one run, reusing matching idempotency keys atomically."""
        if state.status is not RunStatus.QUEUED:
            raise ValueError("new persistent runs must be queued")
        now = time.time()
        with self.database.transaction() as connection:
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
                    return decode_run(row)
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
        with self.database.transaction() as connection:
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
        with self.database.transaction() as connection:
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
                    return decode_run(repeated)
            rows = connection.execute(
                """SELECT * FROM runs WHERE investigation_id=?
                   ORDER BY created_at DESC,rowid DESC LIMIT 3""",
                (str(investigation_id),),
            ).fetchall()
            if not rows:
                raise RunNotFound(str(investigation_id))
            parent = decode_run(rows[0])
            if RunStatus(rows[0]["status"]) not in TERMINAL:
                raise RunConflict("finish or cancel the active round first")
            state, evidence = plan_follow_up(
                parent, [RunState.model_validate_json(row["state_json"])
                         for row in reversed(rows)],
                investigation_id=investigation_id, message=message,
                purpose=purpose, limits=limits, allowed_tools=allowed_tools,
                start=start, end=end, memory_policy=memory_policy,
            )
            self._insert_run(connection, state, evidence, request_hash,
                             idempotency_key, now)
        return self.get(state.id)

    def event_snapshot(self, run_id: UUID) -> EventSnapshot:
        """Project persisted events without loading large evidence payloads."""
        with self.database.transaction() as connection:
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
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE id=?", (str(run_id),)
            ).fetchone()
        if row is None:
            raise RunNotFound(str(run_id))
        return decode_run(row)

    def claim_next(self, *, lease_seconds: float = 30) -> ClaimedRun | None:
        """Claim a queued run or a worker's expired lease."""
        now = time.time()
        with self.database.transaction() as connection:
            connection.execute("BEGIN IMMEDIATE")
            while True:
                row = connection.execute(
                    """SELECT * FROM runs WHERE status='queued' OR
                        (status='running' AND lease_until<?)
                       ORDER BY created_at LIMIT 1""", (now,),
                ).fetchone()
                if row is None:
                    return None
                stored = decode_run(row)
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
        with self.database.transaction() as connection:
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
        with self.database.transaction() as connection:
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
        with self.database.transaction() as connection:
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
        with self.database.transaction() as connection:
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
        with self.database.transaction() as connection:
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
                        evidence=decode_run(row).evidence,
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
        with self.database.transaction() as connection:
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
