"""Deterministic diagnosis checks against held-out case expectations."""

from tracemind.agent import DiagnosisRun
from tracemind.models import DiagnosisStatus
from tracemind.evaluation.models import EvaluationCase, EvaluationResult


def evaluate_diagnosis(
    case: EvaluationCase,
    run: DiagnosisRun,
    *,
    elapsed_ms: float,
) -> EvaluationResult:
    """Score a run using deterministic case checks without model calls.

    Match service and root-cause terms by case-insensitive substring,
    verify citation membership and required source/service, and check
    tool usage. Passing requires the expected diagnosis status; complete
    cases additionally require the cause and source checks. These heuristics
    do not prove causality; elapsed_ms is supplied by the caller.
    """
    diagnosis = run.diagnosis
    normalized_root_cause = diagnosis.root_cause.casefold()
    expected_service = case.ground_truth.root_cause_service.casefold()
    service_match = expected_service in normalized_root_cause
    term_match = any(
        term.casefold() in normalized_root_cause
        for term in case.ground_truth.root_cause_terms
    )

    evidence_by_id = {item.id: item for item in run.evidence}
    grounded = all(
        item_id in evidence_by_id for item_id in diagnosis.evidence_ids
    )
    cited_evidence = [
        evidence_by_id[item_id]
        for item_id in diagnosis.evidence_ids
        if item_id in evidence_by_id
    ]
    required_evidence_match = any(
        item.source is case.ground_truth.required_evidence_source
        and item.service is not None
        and item.service.casefold() == expected_service
        for item in cited_evidence
    )
    within_tool_budget = run.tool_call_count <= case.max_tool_calls
    correct_cause = all((service_match, term_match,
                         required_evidence_match, grounded))
    false_completion = (
        diagnosis.status is DiagnosisStatus.COMPLETE
        and (case.ground_truth.expected_status is not DiagnosisStatus.COMPLETE
             or not correct_cause)
    )
    passed = all(
        (
            diagnosis.status is case.ground_truth.expected_status,
            (correct_cause if case.ground_truth.expected_status
             is DiagnosisStatus.COMPLETE else True),
            grounded,
            within_tool_budget,
        )
    )

    return EvaluationResult(
        case_id=case.id,
        diagnosis_status=diagnosis.status,
        root_cause_service_match=service_match,
        root_cause_term_match=term_match,
        required_evidence_match=required_evidence_match,
        evidence_grounded=grounded,
        tool_call_count=run.tool_call_count,
        within_tool_budget=within_tool_budget,
        elapsed_ms=elapsed_ms,
        passed=passed,
        false_completion=false_completion,
    )
