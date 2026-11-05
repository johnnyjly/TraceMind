"""Human triage preserves useful statements without false verification."""

from uuid import uuid4

import pytest

from tracemind.agent.workflow.triage import reviewed_draft, unreviewed_draft
from tracemind.agent.workflow.validation import render_items
from tracemind.models.diagnosis import Diagnosis, DiagnosisStatus
from tracemind.models.verification import (
    Claim, CompletionAssessment, DiagnosisReview, ReviewItem,
)


def sample():
    evidence = uuid4()
    items = [
        ReviewItem(id="F1", section="finding", text="Payment Charge failed.",
                   evidence_ids=[evidence]),
        ReviewItem(id="H1", section="hypothesis",
                   text="The deeper mechanism remains unknown."),
        ReviewItem(id="P1", section="proposed_check",
                   text="Inspect which token validation caused the error.",
                   evidence_ids=[evidence]),
        ReviewItem(id="P2", section="proposed_check",
                   text="Read the payment source at the reported line.",
                   evidence_ids=[evidence]),
    ]
    candidate = render_items(items, status="complete", confidence=.85)
    return candidate, items, evidence


def assessed(items, evidence, *, completed=True, premise="insufficient"):
    return DiagnosisReview(
        claims=[
            Claim(item_id="F1", text=items[0].text, kind="diagnostic",
                  verdict="supported", supporting_ids=[evidence],
                  rationale="The retained Charge span reports an error."),
            Claim(item_id="H1", text=items[1].text, kind="uncertainty",
                  verdict="supported", missing_observations=["Source code"],
                  rationale="Telemetry does not reveal the implementation."),
            Claim(item_id="P1", text=items[2].text, kind="diagnostic",
                  verdict=premise,
                  supporting_ids=[evidence] if premise == "supported" else [],
                  conflicting_ids=([evidence]
                                   if premise == "contradicted" else []),
                  missing_observations=(["Proof of token validation"]
                                        if premise == "insufficient" else []),
                  rationale="Assess the proposal's factual premise."),
            Claim(item_id="P2", text=items[3].text, kind="proposal",
                  verdict="supported", rationale="A future inspection."),
        ],
        uncertainty_appropriate=True, root_cause_correctness=None,
        rationale="Item-addressed review.",
        completion=[CompletionAssessment(
            criterion_id="actionable_finding", met=completed,
            item_ids=["F1"] if completed else [],
            missing_requirements=[] if completed else ["Failure location"],
            rationale="The scoped initial investigation objective.",
        )],
    )


def test_ready_triage_retains_supported_fact_and_qualified_checks():
    candidate, items, evidence = sample()
    before = candidate.model_dump_json(), [i.model_dump_json() for i in items]
    result = reviewed_draft(candidate, items, assessed(items, evidence))
    assert result.triage_status == "ready"
    assert result.status is DiagnosisStatus.INCONCLUSIVE
    assert result.review_status == "completed"
    assert [i.id for i in result.items] == [i.id for i in items]
    assert result.items[0].evidence_support == "supported"
    assert result.items[2].evidence_support == "insufficient"
    assert result.items[3].evidence_support == "not_applicable"
    assert len(result.remediation) == 2
    assert "insufficient evidence" in result.remediation[0]
    assert "not an observed incident fact" in result.remediation[1]
    assert "hypothesis not confirmed" in result.root_cause
    assert result.confidence == candidate.confidence
    assert "uncalibrated" in result.root_cause
    assert result.feedback_updates == []
    assert before == (candidate.model_dump_json(),
                      [i.model_dump_json() for i in items])


def test_contradicted_item_and_counterevidence_stay_visible():
    candidate, items, evidence = sample()
    review = assessed(items, evidence, premise="contradicted")
    result = reviewed_draft(candidate, items, review)
    item = result.items[2]
    assert item.evidence_support == "contradicted"
    assert item.conflicting_evidence_ids == [evidence]
    assert item.text == items[2].text
    assert "conflicting evidence" in result.remediation[0]
    assert result.status is DiagnosisStatus.INCONCLUSIVE
    assert result.triage_status == "ready"


def test_complete_status_still_requires_acceptance_and_completion():
    candidate, items, evidence = sample()
    review = assessed(items, evidence, premise="supported")
    result = reviewed_draft(candidate, items, review)
    assert result.status is DiagnosisStatus.COMPLETE
    assert result.triage_status == "ready"
    review.completion[0].met = False
    review.completion[0].missing_requirements = ["Failure location"]
    result = reviewed_draft(candidate, items, review)
    assert result.status is DiagnosisStatus.INCONCLUSIVE
    assert result.triage_status == "partial"
    assert ("requirements not established: actionable_finding"
            in result.root_cause)


@pytest.mark.parametrize("error,status", [
    ("timeout", "timed_out"), ("deadline", "timed_out"),
    ("model_unavailable", "unavailable"), (None, "not_requested"),
])
def test_failed_review_does_not_fabricate_item_support(error, status):
    candidate, items, _ = sample()
    result = unreviewed_draft(candidate, items, error)
    assert result.status is DiagnosisStatus.INCONCLUSIVE
    assert result.review_status == status
    assert result.review_error_code == error
    assert result.triage_status == "partial"
    assert result.evidence_ids == candidate.evidence_ids
    for item in result.items:
        assert item.review_status == "unreviewed"
        assert item.evidence_support == "not_assessed"
        assert item.kind is None and item.rationale is None
        assert not item.supporting_evidence_ids
        assert not item.conflicting_evidence_ids
    assert all("UNREVIEWED" in check for check in result.remediation)
    assert "[F1] [UNREVIEWED" in result.root_cause
    assert result.feedback_updates == []


def test_process_review_is_not_a_confirmed_incident_fact():
    candidate, items, evidence = sample()
    review = assessed(items, evidence)
    review.claims[0].kind = "process"
    review.claims[0].supporting_ids = []
    review.claims[0].supporting_event_sequences = [17]
    review.completion = None
    result = reviewed_draft(candidate, items, review)
    assert result.items[0].evidence_support == "not_applicable"
    assert result.triage_status == "no_findings"
    assert "not an incident conclusion" in result.root_cause


def test_new_delivery_fields_round_trip_in_public_diagnosis():
    candidate, items, _ = sample()
    draft = unreviewed_draft(candidate, items, "timeout")
    diagnosis = Diagnosis(incident_id=uuid4(), **draft.model_dump(
        exclude={"feedback_updates"}
    ))
    restored = Diagnosis.model_validate_json(diagnosis.model_dump_json())
    assert restored.items == draft.items
    assert restored.review_status == "timed_out"
    assert restored.triage_status == "partial"


def test_no_findings_does_not_hide_a_useful_next_check():
    candidate, items, _ = sample()
    checks = [items[-1]]
    result = unreviewed_draft(candidate, checks, "timeout")
    assert result.triage_status == "no_findings"
    assert result.items[0].section == "proposed_check"
    assert len(result.remediation) == 1
    assert "UNREVIEWED" in result.remediation[0]


def test_rpc_success_and_business_outcome_keep_independent_support():
    rpc = uuid4()
    items = [
        ReviewItem(id="F1", section="finding",
                   text="Charge RPC returned OK.", evidence_ids=[rpc]),
        ReviewItem(id="F2", section="finding",
                   text="Payment funds settled.", evidence_ids=[rpc]),
        ReviewItem(id="P1", section="proposed_check",
                   text="Check linked transaction state and idempotency "
                   "before retry."),
    ]
    candidate = render_items(items, status="complete", confidence=.9)
    result = DiagnosisReview(
        claims=[
            Claim(item_id="F1", text=items[0].text, kind="diagnostic",
                  verdict="supported", supporting_ids=[rpc],
                  rationale="The protocol status is explicit."),
            Claim(item_id="F2", text=items[1].text, kind="diagnostic",
                  verdict="insufficient",
                  missing_observations=["Linked settlement/transaction state"],
                  rationale="RPC success alone does not establish "
                  "settlement."),
            Claim(item_id="P1", text=items[2].text, kind="proposal",
                  verdict="supported", rationale="Future transaction check."),
        ], uncertainty_appropriate=True, root_cause_correctness=None,
        rationale="Scripted labels exercise rendering, not model accuracy.",
    )
    delivered = reviewed_draft(candidate, items, result)
    assert [item.evidence_support for item in delivered.items] == [
        "supported", "insufficient", "not_applicable",
    ]
    assert delivered.items[1].text == items[1].text
    assert delivered.items[1].missing_observations == [
        "Linked settlement/transaction state",
    ]
    assert "[F2] [REVIEWED; insufficient evidence]" in delivered.root_cause
    assert "idempotency" in delivered.remediation[0]
