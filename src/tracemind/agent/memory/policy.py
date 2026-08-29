"""Memory provenance and feedback planning, independent of SQL transactions."""

from datetime import datetime, timedelta
from uuid import UUID, uuid4

from tracemind.models.memory import (
    FeedbackUpdate,
    GuidanceRecord,
    MemoryCreate,
    MemoryRecord,
)
from tracemind.models.run import RunState, RunStatus
from tracemind.models.storage import RunConflict, StoredRun


def confirmed_record(
    content: MemoryCreate, run: StoredRun, *, memory_id: UUID | None,
    revision: int, now: datetime,
) -> MemoryRecord:
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
    return record


def plan_guidance(
    state: RunState, update: FeedbackUpdate, current: GuidanceRecord | None, *,
    now: datetime,
) -> GuidanceRecord | None:
    current = current.model_copy(deep=True) if current is not None else None
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
            current.updated_at = now
        return current
    record = GuidanceRecord(
        id=current.id if current else uuid4(), key=update.key,
        kind=update.kind, instruction=update.content,
        revision=current.revision + 1 if current else 1,
        source_run_id=state.id, source_quote=update.quote,
        rationale=update.rationale, updated_at=now,
    )
    return record


def plan_diagnosis_memory(
    state: RunState, update: FeedbackUpdate, source: StoredRun,
    current: MemoryRecord | None, *, now: datetime,
) -> MemoryRecord | None:
    current = current.model_copy(deep=True) if current is not None else None
    if source.state.incident.id != state.incident.id:
        raise RunConflict("diagnosis_source_incident_mismatch")
    scope = state.learning_policy.scope
    source_scope = (source.state.learning_policy or
                    source.state.memory_policy)
    if source_scope is not None and source_scope.scope != scope:
        raise RunConflict("diagnosis_source_scope_mismatch")
    if current is not None and current.updated_at > state.started_at:
        raise RunConflict("diagnosis_revision_conflict")
    if update.action == "delete":
        if current is not None:
            current.status = "revoked"
            current.revision += 1
            current.updated_at = now
            current.feedback_run_id = state.id
            current.feedback_quote = update.quote
        return current
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
    return record
