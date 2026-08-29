"""Completion checks independent of storage and model execution."""
from collections import Counter
from datetime import datetime

from tracemind.agent.errors import DiagnosisValidationError
from tracemind.models import DiagnosisStatus, Evidence
from tracemind.models.result import DiagnosisDraft


def validate_draft(
    draft: DiagnosisDraft,
    evidence: list[Evidence],
    *,
    start: datetime,
    end: datetime,
) -> None:
    """Reject unsupported citations, blank explanations, and invalid
    times.

    A complete result must cite evidence; all citations must be unique
    and refer to observations in the inclusive investigation window.
    These checks establish provenance and time relevance, not causal
    correctness.
    """
    observed = {item.id: item for item in evidence}
    unknown = list(dict.fromkeys(
        i for i in draft.evidence_ids if i not in observed
    ))
    if unknown:
        raise DiagnosisValidationError(
            "agent cited evidence unavailable in this run",
            code="unknown_evidence", evidence_ids=unknown,
        )
    duplicates = [i for i, count in Counter(draft.evidence_ids).items()
                  if count > 1]
    if duplicates:
        raise DiagnosisValidationError(
            "diagnosis contains duplicate evidence references",
            code="duplicate_evidence", evidence_ids=duplicates,
        )
    if not draft.root_cause.strip():
        raise DiagnosisValidationError(
            "diagnosis must explain the result", code="blank_root_cause",
        )
    if draft.status is DiagnosisStatus.COMPLETE and not draft.evidence_ids:
        raise DiagnosisValidationError(
            "a complete diagnosis must cite available evidence",
            code="missing_evidence",
        )
    outside = []
    for evidence_id in draft.evidence_ids:
        observed_at = observed[evidence_id].observed_at
        if (
            observed_at.utcoffset() is None
            or not start <= observed_at <= end
        ):
            outside.append(evidence_id)
    if outside:
        raise DiagnosisValidationError(
            "diagnosis cites evidence outside the investigation window "
            f"[{start.isoformat()}, {end.isoformat()}]",
            code="evidence_outside_window", evidence_ids=outside,
        )
