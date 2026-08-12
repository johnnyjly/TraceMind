from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from tracemind.investigation import DeterministicVerifier, EvidenceContext
from tracemind.models import (
    Evidence,
    EvidenceLink,
    EvidenceRelation,
    EvidenceSource,
    Hypothesis,
    Incident,
    InvestigationState,
    VerificationOutcome,
    VerificationDecision,
)


NOW = datetime(2026, 9, 23, tzinfo=UTC)


def make_evidence(
    source: EvidenceSource,
    summary: str,
    *,
    service: str = "payment",
    seconds: int = 0,
) -> Evidence:
    return Evidence(
        source=source,
        summary=summary,
        observed_at=NOW + timedelta(seconds=seconds),
        service=service,
    )


def test_investigation_state_rejects_unknown_evidence_link() -> None:
    hypothesis = Hypothesis(
        statement="Payment charge fails",
        suspected_service="payment",
    )

    with pytest.raises(ValidationError, match="unknown evidence"):
        InvestigationState(
            incident=Incident(title="Checkout failure", description="HTTP 500"),
            hypotheses=[hypothesis],
            evidence_links=[
                EvidenceLink(
                    hypothesis_id=hypothesis.id,
                    evidence_id=uuid4(),
                    relation=EvidenceRelation.SUPPORTS,
                    rationale="The charge span failed.",
                )
            ],
        )


def test_verification_decision_requires_grounding_and_next_actions() -> None:
    with pytest.raises(ValidationError, match="supporting evidence"):
        VerificationDecision(
            outcome=VerificationOutcome.SUFFICIENT,
            reason="Enough evidence.",
        )

    with pytest.raises(ValidationError, match="next action"):
        VerificationDecision(
            outcome=VerificationOutcome.CONTINUE,
            reason="More evidence is required.",
        )


def test_evidence_context_deduplicates_repeated_observations() -> None:
    older = make_evidence(
        EvidenceSource.LOG,
        "Payment request failed",
        seconds=1,
    )
    newer = make_evidence(
        EvidenceSource.LOG,
        "  PAYMENT request   failed ",
        seconds=2,
    )

    result = EvidenceContext().deduplicate([older, newer])

    assert len(result) == 1
    assert result[0].id == newer.id
    assert result[0].attributes["duplicate_count"] == 2


def test_evidence_context_prioritizes_links_and_suspected_service() -> None:
    hypothesis = Hypothesis(
        statement="Payment charge fails",
        suspected_service="payment",
    )
    payment_trace = make_evidence(EvidenceSource.TRACE, "Charge failed", seconds=1)
    checkout_log = make_evidence(
        EvidenceSource.LOG,
        "Checkout returned 500",
        service="checkout",
        seconds=3,
    )
    payment_log = make_evidence(
        EvidenceSource.LOG,
        "Payment token rejected",
        seconds=2,
    )
    links = [
        EvidenceLink(
            hypothesis_id=hypothesis.id,
            evidence_id=payment_trace.id,
            relation=EvidenceRelation.SUPPORTS,
            rationale="The payment span is an error.",
        )
    ]

    result = EvidenceContext(max_items_per_hypothesis=2).select(
        hypothesis,
        [checkout_log, payment_log, payment_trace],
        links,
    )

    assert [item.id for item in result] == [payment_trace.id, payment_log.id]


def test_evidence_context_preserves_linked_duplicate() -> None:
    hypothesis = Hypothesis(
        statement="Payment charge fails",
        suspected_service="payment",
    )
    linked = make_evidence(EvidenceSource.LOG, "Payment request failed", seconds=1)
    newer = make_evidence(EvidenceSource.LOG, "Payment request failed", seconds=2)
    links = [
        EvidenceLink(
            hypothesis_id=hypothesis.id,
            evidence_id=linked.id,
            relation=EvidenceRelation.SUPPORTS,
            rationale="This log was cited by the investigator.",
        )
    ]

    result = EvidenceContext().select(hypothesis, [linked, newer], links)

    assert len(result) == 1
    assert result[0].id == linked.id
    assert result[0].attributes["duplicate_count"] == 2


def test_verifier_accepts_independent_grounded_evidence() -> None:
    hypothesis = Hypothesis(
        statement="Payment charge rejects the token",
        suspected_service="payment",
    )
    trace = make_evidence(EvidenceSource.TRACE, "Payment charge span failed")
    log = make_evidence(EvidenceSource.LOG, "Payment request failed", seconds=1)
    state = InvestigationState(
        incident=Incident(title="Checkout failure", description="HTTP 500"),
        hypotheses=[hypothesis],
        evidence=[trace, log],
        evidence_links=[
            EvidenceLink(
                hypothesis_id=hypothesis.id,
                evidence_id=item.id,
                relation=EvidenceRelation.SUPPORTS,
                rationale="Direct payment failure evidence.",
            )
            for item in (trace, log)
        ],
        round_number=1,
        tool_calls=3,
    )

    decision = DeterministicVerifier().verify(state)

    assert decision.outcome is VerificationOutcome.SUFFICIENT
    assert decision.selected_hypothesis_id == hypothesis.id
    assert set(decision.supporting_evidence_ids) == {trace.id, log.id}


def test_verifier_requests_targeted_follow_up_when_evidence_is_thin() -> None:
    hypothesis = Hypothesis(
        statement="Payment charge rejects the token",
        suspected_service="payment",
    )
    trace = make_evidence(EvidenceSource.TRACE, "Payment charge span failed")
    state = InvestigationState(
        incident=Incident(title="Checkout failure", description="HTTP 500"),
        hypotheses=[hypothesis],
        evidence=[trace],
        evidence_links=[
            EvidenceLink(
                hypothesis_id=hypothesis.id,
                evidence_id=trace.id,
                relation=EvidenceRelation.SUPPORTS,
                rationale="The payment span is an error.",
            )
        ],
        round_number=1,
        tool_calls=2,
    )

    decision = DeterministicVerifier().verify(state)

    assert decision.outcome is VerificationOutcome.CONTINUE
    assert decision.next_actions == [
        "Collect more evidence for payment: Payment charge rejects the token"
    ]


def test_verifier_does_not_complete_with_contradictory_evidence() -> None:
    hypothesis = Hypothesis(
        statement="Payment charge rejects the token",
        suspected_service="payment",
    )
    trace = make_evidence(EvidenceSource.TRACE, "Payment charge span failed")
    log = make_evidence(EvidenceSource.LOG, "Payment request failed", seconds=1)
    healthy_metric = make_evidence(
        EvidenceSource.METRIC,
        "Payment success rate is normal",
        seconds=2,
    )
    state = InvestigationState(
        incident=Incident(title="Checkout failure", description="HTTP 500"),
        hypotheses=[hypothesis],
        evidence=[trace, log, healthy_metric],
        evidence_links=[
            EvidenceLink(
                hypothesis_id=hypothesis.id,
                evidence_id=item.id,
                relation=relation,
                rationale="Evidence assessed against the hypothesis.",
            )
            for item, relation in (
                (trace, EvidenceRelation.SUPPORTS),
                (log, EvidenceRelation.SUPPORTS),
                (healthy_metric, EvidenceRelation.CONTRADICTS),
            )
        ],
        round_number=1,
        tool_calls=3,
    )

    decision = DeterministicVerifier().verify(state)

    assert decision.outcome is VerificationOutcome.CONTINUE
    assert decision.contradicting_evidence_ids == [healthy_metric.id]
    assert "Resolve contradictory evidence" in decision.next_actions[0]


def test_verifier_stops_when_budget_is_exhausted() -> None:
    hypothesis = Hypothesis(
        statement="Payment charge rejects the token",
        suspected_service="payment",
    )
    state = InvestigationState(
        incident=Incident(title="Checkout failure", description="HTTP 500"),
        hypotheses=[hypothesis],
        round_number=2,
        max_rounds=2,
        tool_calls=12,
        max_tool_calls=12,
    )

    decision = DeterministicVerifier().verify(state)

    assert decision.outcome is VerificationOutcome.INCONCLUSIVE
    assert decision.next_actions == []
