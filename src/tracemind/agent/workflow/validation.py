"""Completion checks independent of storage and model execution."""
from collections import Counter
from collections.abc import Sequence
from datetime import datetime

from tracemind.agent.errors import DiagnosisValidationError
from tracemind.models import DiagnosisStatus, Evidence
from tracemind.models.result import DiagnosisDraft
from tracemind.models.knowledge import KnowledgeReference


def validate_draft(
    draft: DiagnosisDraft,
    evidence: list[Evidence],
    *,
    start: datetime,
    end: datetime,
    knowledge: Sequence[KnowledgeReference] = (),
) -> None:
    """Reject unsupported citations, blank explanations, and invalid
    times.

    A complete result must cite evidence; all citations must be unique
    and refer to observations in the inclusive investigation window.
    These checks establish provenance and time relevance, not causal
    correctness.
    """
    observed = {item.id: item for item in evidence}
    known_refs = {item.id for item in knowledge}
    unknown_refs = [ref for ref in draft.knowledge_refs
                    if ref not in known_refs]
    if unknown_refs:
        raise DiagnosisValidationError(
            "agent cited knowledge unavailable in this run",
            code="unknown_knowledge",
            knowledge_refs=unknown_refs,
        )
    if len(set(draft.knowledge_refs)) != len(draft.knowledge_refs):
        raise DiagnosisValidationError(
            "diagnosis contains duplicate knowledge references",
            code="duplicate_knowledge",
            knowledge_refs=draft.knowledge_refs,
        )
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


def resolve_proposal(proposal, state, evidence):
    """Resolve typed references and render only the submitted statements."""
    from uuid import UUID
    from tracemind.agent.workflow.references import register_references
    from tracemind.models.verification import ReviewItem

    register_references(state)
    sections = (("findings", "finding", "F"),
                ("hypotheses", "hypothesis", "H"),
                ("limitations", "limitation", "L"),
                ("proposed_checks", "proposed_check", "P"))
    items = []
    for field, section, prefix in sections:
        for number, statement in enumerate(getattr(proposal, field), 1):
            refs = statement.evidence_refs + statement.knowledge_refs
            unknown = [ref for ref in refs if ref not in state.reference_map]
            wrong = ([ref for ref in statement.evidence_refs
                      if not ref.startswith("E")]
                     + [ref for ref in statement.knowledge_refs
                        if not ref.startswith("K")])
            if unknown or wrong:
                raise DiagnosisValidationError(
                    "Use only available E references for telemetry and K "
                    "references for knowledge.", code="unknown_reference",
                    references=list(dict.fromkeys(unknown + wrong)),
                )
            if len(refs) != len(set(refs)):
                raise DiagnosisValidationError(
                    "Statement references must be unique.",
                    code="duplicate_reference",
                    references=list(dict.fromkeys(refs)),
                )
            items.append(ReviewItem(
                id=f"{prefix}{number}", section=section, text=statement.text,
                evidence_ids=[UUID(state.reference_map[ref])
                              for ref in statement.evidence_refs],
                knowledge_refs=[state.reference_map[ref]
                                for ref in statement.knowledge_refs],
            ))
    if proposal.status is DiagnosisStatus.COMPLETE and not proposal.findings:
        raise DiagnosisValidationError(
            "A complete proposal needs an observed finding.",
            code="missing_finding")
    if not items or not any(i.text.strip() for i in items):
        raise DiagnosisValidationError(
            "Return findings or specific limitations.", code="empty_proposal")
    draft = render_items(items, proposal.status, proposal.confidence)
    draft.feedback_updates = proposal.feedback_updates
    validate_draft(draft, evidence, start=state.investigation_start,
                   end=state.investigation_end, knowledge=state.knowledge)
    state.candidate_items = items
    return draft


def render_items(items, status="inconclusive", confidence=0):
    """No extra model-authored prose can bypass item review."""
    labels = {"finding": "Observed findings", "hypothesis": "Hypotheses",
              "limitation": "Scope and unknowns"}
    parts = []
    for section, label in labels.items():
        selected = [i for i in items if i.section == section]
        if selected:
            parts.append(label + ":\n" + "\n".join(
                f"- [{i.id}] {i.text}" for i in selected))
    return DiagnosisDraft(
        status=status, confidence=confidence,
        root_cause=("\n\n".join(parts)
                    or "No incident finding was established."),
        evidence_ids=list(dict.fromkeys(
            r for i in items for r in i.evidence_ids)),
        knowledge_refs=list(dict.fromkeys(
            r for i in items for r in i.knowledge_refs)),
        remediation=[i.text for i in items if i.section == "proposed_check"],
    )
