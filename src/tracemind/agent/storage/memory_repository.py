"""Transactional incident memory and human feedback.

Shares the run database.
"""

from __future__ import annotations

from datetime import UTC, datetime
import sqlite3
from uuid import UUID

from pydantic import ValidationError

from tracemind.agent.storage.database import RunDatabase
from tracemind.agent.memory.policy import (
    confirmed_record,
    plan_diagnosis_memory,
    plan_guidance,
)
from tracemind.agent.storage.run_codec import decode_run
from tracemind.models import Incident
from tracemind.models.memory import (
    FeedbackResult,
    FeedbackUpdate,
    GuidanceRecord,
    GuidanceRevision,
    GuidanceSnapshot,
    MemoryCreate,
    MemoryPolicy,
    MemoryRecord,
    MemorySnapshot,
)
from tracemind.models.run import RunState
from tracemind.models.storage import RunConflict, RunNotFound


class MemoryNotFound(KeyError):
    """The historical experience has been deleted or never existed."""


class MemoryRepository:
    """Manage human confirmations using the existing local run database.

    Workflow validates feedback proposals under the run lease. Revisions use
    optimistic concurrency;
    previously captured run snapshots retain the exact historical content.
    """

    def __init__(self, database: RunDatabase) -> None:
        """Reuse SQLite transactions and source-run provenance checks."""
        self.database = database

    def get(self, memory_id: UUID) -> MemoryRecord:
        """Read current content, including revoked or expired records."""
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT record_json FROM incident_memories WHERE id=?",
                (str(memory_id),),
            ).fetchone()
        if row is None:
            raise MemoryNotFound(str(memory_id))
        return MemoryRecord.model_validate_json(row[0])

    def list(self, *, offset: int = 0, limit: int = 20) -> list[MemoryRecord]:
        """Page current records for manual review, without hiding expiry."""
        with self.database.transaction() as connection:
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
        with self.database.transaction() as connection:
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
            run = decode_run(source)
            record = confirmed_record(
                content, run, memory_id=memory_id, revision=revision, now=now,
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
        with self.database.transaction() as connection:
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
        with self.database.transaction() as connection:
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
            with self.database.transaction() as connection:
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
        with self.database.transaction() as connection:
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
        with self.database.transaction() as connection:
            rows = connection.execute(
                "SELECT record_json FROM agent_guidance ORDER BY key "
                "LIMIT ? OFFSET ?", (limit, offset),
            ).fetchall()
        return [GuidanceRecord.model_validate_json(row[0]) for row in rows]

    def revise_guidance(
        self, guidance_id: UUID, revision: GuidanceRevision,
    ) -> GuidanceRecord:
        """Correct a learned instruction; preserve old run snapshots."""
        with self.database.transaction() as connection:
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
        with self.database.transaction() as connection:
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
        with self.database.transaction() as connection:
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
        record = plan_guidance(state, update, current, now=datetime.now(UTC))
        if record is None:
            return None
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
        source = decode_run(row)
        scope = state.learning_policy.scope
        prior = connection.execute(
            """SELECT record_json FROM incident_memories
               WHERE source_incident_id=? AND environment=?
               AND deployment_version=? ORDER BY updated_at DESC LIMIT 1""",
            (str(state.incident.id), scope.environment,
             scope.deployment_version),
        ).fetchone()
        current = MemoryRecord.model_validate_json(prior[0]) if prior else None
        record = plan_diagnosis_memory(
            state, update, source, current, now=datetime.now(UTC),
        )
        if record is None:
            return None
        connection.execute(
            """INSERT INTO incident_memories VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET revision=excluded.revision,
               status=excluded.status,confirmed_at=excluded.confirmed_at,
               expires_at=excluded.expires_at,updated_at=excluded.updated_at,
               record_json=excluded.record_json""",
            (str(record.id), record.revision, record.environment,
             record.deployment_version, str(record.source_incident_id),
             record.status, record.confirmed_at.timestamp(),
             record.expires_at.timestamp(),
             record.updated_at.timestamp(), record.model_dump_json()),
        )
        return record.id
