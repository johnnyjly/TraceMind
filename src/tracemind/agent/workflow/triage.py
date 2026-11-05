"""Deliver initial triage without promoting unreviewed claims to facts.

Callers validate candidate provenance and the review contract before delivery.
Raw candidate rendering and review fingerprints remain untouched.
"""

from tracemind.models.diagnosis import DiagnosisStatus, TriageItem
from tracemind.models.result import DiagnosisDraft


def _label(item):
    if item.review_status == "unreviewed":
        return "UNREVIEWED; evidence support not assessed"
    if item.evidence_support == "insufficient":
        return "REVIEWED; insufficient evidence"
    if item.evidence_support == "contradicted":
        return "REVIEWED; conflicting evidence"
    if item.kind == "proposal":
        return "REVIEWED PROPOSED CHECK; not an observed incident fact"
    if item.kind == "process":
        return "REVIEWED PROCESS STATEMENT; not an incident conclusion"
    if item.kind == "uncertainty":
        return "REVIEWED UNCERTAINTY; hypothesis not confirmed"
    return "REVIEWED; evidence-supported"


def _statement(item):
    return f"[{item.id}] [{_label(item)}] {item.text}"


def _delivery(candidate, items, *, status, triage_status, review_status,
              error_code=None, completion_gaps=()):
    heading = (
        f"Initial triage. Review status: {review_status}. "
        "Legacy confidence is an uncalibrated model estimate."
    )
    if error_code:
        heading += f" Review issue: {error_code}."
    if completion_gaps:
        heading += (" Initial-triage requirements not established: "
                    + ", ".join(completion_gaps) + ".")
    sections = (
        ("finding", "Findings and leads"),
        ("hypothesis", "Hypotheses"),
        ("limitation", "Scope and unknowns"),
        ("proposed_check", "Proposed checks"),
    )
    parts = [heading]
    for section, label in sections:
        selected = [item for item in items if item.section == section]
        if selected:
            parts.append(label + ":\n" + "\n".join(
                "- " + _statement(item) for item in selected
            ))
    if not items:
        parts.append("No incident finding was retained.")
    return DiagnosisDraft(
        status=status, root_cause="\n\n".join(parts),
        confidence=candidate.confidence,
        evidence_ids=list(dict.fromkeys(
            ref for item in items for ref in (
                item.evidence_ids + item.supporting_evidence_ids
                + item.conflicting_evidence_ids
            )
        )),
        knowledge_refs=list(dict.fromkeys(
            ref for item in items for ref in item.knowledge_refs
        )),
        remediation=[_statement(item) for item in items
                     if item.section == "proposed_check"],
        items=items, review_status=review_status,
        review_error_code=error_code, triage_status=triage_status,
        feedback_updates=[],
    )


def reviewed_draft(candidate, items, review):
    """Retain every assessed item, including qualified next checks."""
    claims = {claim.item_id: claim for claim in review.claims}
    delivered = []
    for item in items:
        claim = claims[item.id]
        support = claim.verdict
        if support == "supported" and claim.kind in {"proposal", "process"}:
            support = "not_applicable"
        delivered.append(TriageItem(
            **item.model_dump(), review_status="reviewed", kind=claim.kind,
            evidence_support=support,
            supporting_evidence_ids=claim.supporting_ids,
            conflicting_evidence_ids=claim.conflicting_ids,
            supporting_event_sequences=claim.supporting_event_sequences,
            conflicting_event_sequences=claim.conflicting_event_sequences,
            missing_observations=claim.missing_observations,
            rationale=claim.rationale,
        ))
    has_findings = any(
        claim.kind == "diagnostic" and claim.verdict == "supported"
        for claim in review.claims
    )
    return _delivery(
        candidate, delivered,
        status=(DiagnosisStatus.COMPLETE
                if review.accepted and review.task_completed
                else DiagnosisStatus.INCONCLUSIVE),
        triage_status=("ready" if review.task_completed
                       else "partial" if has_findings else "no_findings"),
        review_status="completed",
        completion_gaps=[assessment.criterion_id
                         for assessment in review.completion or []
                         if not assessment.met],
    )


def unreviewed_draft(candidate, items, error_code, *, pending=False):
    """Publish a provenance-checked candidate with no semantic endorsement."""
    delivered = [TriageItem(
        **item.model_dump(), review_status="unreviewed",
        evidence_support="not_assessed",
    ) for item in items]
    return _delivery(
        candidate, delivered, status=DiagnosisStatus.INCONCLUSIVE,
        triage_status=(
            "partial" if any(item.section == "finding" for item in items)
            else "no_findings"
        ),
        review_status=("pending" if pending and not error_code else
                       "timed_out" if error_code in {"timeout", "deadline"}
                       else "unavailable" if error_code else "not_requested"),
        error_code=error_code,
    )


def current_triage(state, evidence, *, cancel_requested=False):
    """Project only the current, provenance-valid candidate and its review.

    No archived collaboration review can become the current initial result.
    The fingerprint binds the review to candidate, evidence and query history.
    """
    from tracemind.agent.errors import DiagnosisValidationError
    from tracemind.agent.verification import (
        review_items, review_still_current, subject_for, validate_review,
    )
    from tracemind.agent.workflow.validation import validate_draft
    from tracemind.models.diagnosis import Diagnosis
    from tracemind.models.storage import TERMINAL

    record = state.verification
    if record is None or record.candidate.incident_id != state.incident.id:
        return None
    if state.candidate_items and state.candidate_items != record.items:
        return None
    subject = subject_for(state, record.candidate, evidence).model_copy(
        update={"items": record.items},
    )
    raw = DiagnosisDraft(**record.candidate.model_dump(
        exclude={"incident_id"},
    ))
    try:
        validate_draft(raw, evidence, start=subject.start, end=subject.end,
                       knowledge=state.knowledge)
        if review_items(subject) != record.items:
            return None
        if not review_still_current(state, record, evidence):
            return None
    except (DiagnosisValidationError, ValueError):
        return None
    if record.status == "completed" and record.review is not None:
        try:
            validate_review(record.review, subject)
        except ValueError:
            draft = unreviewed_draft(record.candidate, record.items,
                                     "invalid_review")
        else:
            draft = reviewed_draft(
                record.candidate, record.items, record.review,
            )
    else:
        error = record.error_code
        if not error and record.status == "failed":
            error = "review_unavailable"
        if not error and cancel_requested:
            error = "cancelled"
        if not error and state.status in TERMINAL:
            error = (state.stop_reason.value if state.stop_reason
                     else "review_interrupted")
        draft = unreviewed_draft(record.candidate, record.items, error,
                                 pending=error is None)
    return Diagnosis(incident_id=state.incident.id, **draft.model_dump(
        exclude={"feedback_updates"},
    ))
