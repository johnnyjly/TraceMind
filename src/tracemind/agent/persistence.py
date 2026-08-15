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
from tracemind.models import Evidence
from tracemind.models.run import RunState, RunStatus, StopReason


SCHEMA_VERSION = 1
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
            if version not in (0, SCHEMA_VERSION):
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
                connection.execute("PRAGMA user_version=1")

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
            connection.execute(
                """INSERT INTO runs
                   (id,status,state_json,evidence_json,request_hash,
                    idempotency_key,created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (str(state.id), state.status.value, state.model_dump_json(),
                 "[]", request_hash, idempotency_key, now, now),
            )
        return self.get(state.id)

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
                    """SELECT * FROM runs WHERE cancel_requested=0 AND
                       (status='queued' OR
                        (status='running' AND lease_until<?))
                       ORDER BY created_at LIMIT 1""", (now,),
                ).fetchone()
                if row is None:
                    return None
                stored = self._decode(row)
                state = stored.state
                if state.status in TERMINAL:
                    diagnosis = state.diagnosis
                    result = (
                        DiagnosisRun(
                            diagnosis=diagnosis, evidence=stored.evidence,
                            tool_calls=state.tool_calls, state=state,
                        ).model_dump_json() if diagnosis else None
                    )
                    connection.execute(
                        """UPDATE runs SET status=?,result_json=?,
                           lease_owner=NULL,lease_until=NULL,updated_at=?
                           WHERE id=?""",
                        (state.status.value, result, now, row["id"]),
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
                if row["status"] == "queued" or (
                    row["lease_until"] is not None
                    and row["lease_until"] < time.time()
                ):
                    state = RunState.model_validate_json(row["state_json"])
                    state.status = RunStatus.CANCELLED
                    state.stop_reason = StopReason.CANCELLED
                    state.finished_at = datetime.now(UTC)
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
