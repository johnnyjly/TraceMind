"""Versioned code evaluators registered with the Langfuse experiment runner."""

from math import isfinite, log2

from langfuse import Evaluation
from tracemind.agent.runtime.audit import trace_issues
from tracemind.evaluation.models import ExperimentOutput
from tracemind.models import DiagnosisStatus

VERSION = "tracemind-code-v2"


def metric(name, value, comment=None, **metadata):
    return Evaluation(
        name=name,
        value=value,
        comment=comment,
        data_type=(
            "BOOLEAN"
            if isinstance(value, bool)
            else "CATEGORICAL" if isinstance(value, str) else "NUMERIC"
        ),
        metadata={"evaluator_version": VERSION, **metadata},
    )


def finding_scores(diagnosis, review):
    """Count reviewed incident findings actually retained in the answer."""
    findings = []
    if diagnosis is not None and review is not None:
        final_text = " ".join(diagnosis.root_cause.split())
        citations = set(diagnosis.evidence_ids)
        findings = [c for c in review.claims
                    if c.kind == "diagnostic" and c.verdict == "supported"
                    and c.supporting_ids and not c.missing_observations
                    and not c.conflicting_ids
                    and not c.conflicting_event_sequences
                    and set(c.supporting_ids) <= citations
                    and " ".join(c.text.split()) in final_text]
    return [
        metric("grounded_findings", bool(findings),
               "Reviewed diagnostic excerpts retained with their cited "
               "telemetry; not independent root-cause correctness."),
        metric("supported_finding_count", len(findings)),
    ]


def diagnosis_scores(*, output, expected_output=None, **kwargs):
    result = ExperimentOutput.model_validate(output)
    expected = expected_output or {}
    diagnosis = result.diagnosis
    citations = diagnosis.evidence_ids if diagnosis else []
    eligible = {
        e.id
        for e in result.evidence
        if result.investigation_start is not None
        and result.investigation_end is not None
        and result.investigation_start
        <= e.observed_at
        <= result.investigation_end
    }
    citation_valid = (
        len(citations) == len(set(citations)) and set(citations) <= eligible
    )
    if diagnosis and diagnosis.status is DiagnosisStatus.COMPLETE:
        citation_valid = citation_valid and bool(citations)
    state = result.run_state
    scores = [
        metric(
            "execution_success",
            result.error_type is None and diagnosis is not None,
        ),
        metric(
            "citation_integrity",
            citation_valid,
            "Checks reference identity/window only, not semantic support.",
        ),
        metric("elapsed_ms", result.elapsed_ms),
        metric("diagnosis_status", diagnosis.status.value
               if diagnosis else "none"),
    ]
    if state is not None:
        issues = trace_issues(state)
        retries = sum(
            e.kind == "recovery.retry_scheduled" for e in state.events
        )
        scores.extend(
            [
                metric("run_status", state.status.value),
                metric(
                    "stop_reason",
                    state.stop_reason.value if state.stop_reason else "none",
                ),
                metric("backend_attempts", state.backend_attempts),
                metric("retries", retries),
                metric(
                    "recovered",
                    state.status.value == "succeeded" and retries > 0,
                ),
                metric("diagnosis_repairs", state.diagnosis_repairs),
                metric("fallback_activations", state.model_switches),
                metric(
                    "context_compressions",
                    sum(
                        e.kind == "context.selected"
                        and e.data.get("reduced_messages", 0) > 0
                        for e in state.events
                    ),
                ),
                metric(
                    "trace_integrity",
                    not issues,
                    ",".join(issues),
                ),
                metric(
                    "tool_budget",
                    sum(state.tool_calls.values()) <= state.limits.tool_calls,
                ),
                metric("model_calls", state.model_calls),
                metric("tool_calls", sum(state.tool_calls.values())),
                metric(
                    "unreported_model_calls",
                    state.usage.unreported_model_calls,
                ),
                metric("reported_tokens", state.usage.reported_total_tokens),
                metric("cache_reported_model_calls",
                       state.usage.reported_cache_model_calls),
                metric("cache_unreported_model_calls",
                       state.usage.unreported_cache_model_calls),
            ]
        )
        if state.model_calls:
            scores.append(metric(
                "cache_usage_coverage",
                state.usage.reported_cache_model_calls / state.model_calls,
            ))
        if state.usage.reported_cache_model_calls:
            hit = state.usage.reported_cache_hit_tokens
            miss = state.usage.reported_cache_miss_tokens
            scores.extend([
                metric("cache_hit_tokens", hit),
                metric("cache_miss_tokens", miss),
            ])
            if hit + miss:
                scores.append(metric("cache_hit_rate", hit / (hit + miss)))
    if state is not None and state.verification_policy is not None:
        record = state.verification
        review = (record.review if record and record.status == "completed"
                  and citation_valid else None)
        scores.extend(finding_scores(diagnosis, review))
        if diagnosis and diagnosis.status is DiagnosisStatus.COMPLETE:
            scores.append(metric("runtime_verification", bool(
                record and record.status == "completed" and record.review
                and record.review.accepted)))
        if record is not None:
            scores.extend([
                metric("verification_status", record.status),
                metric("verification_completed", record.status == "completed"),
                metric("verification_attempts", record.attempts),
                metric("verification_format_repairs",
                       max(0, record.attempts - 1)),
            ])
            if record.error_code:
                scores.append(metric("verification_error", record.error_code))
            if record.review is not None:
                scores.append(
                    metric(
                        "verification_verdict",
                        record.review.verdict))
    if (
        "required_evidence_source" in expected
        and "root_cause_service" in expected
    ):
        scores.append(
            metric(
                "required_source_citation",
                any(
                    e.id in citations
                    and e.source.value == expected["required_evidence_source"]
                    and (e.service or "").casefold()
                    == expected["root_cause_service"].casefold()
                    for e in result.evidence
                ),
                "Checks citation provenance, not causal correctness.",
            )
        )
    if "expected_status" in expected:
        scores.append(
            metric(
                "expected_status",
                diagnosis is not None
                and diagnosis.status.value == expected["expected_status"],
            )
        )
    if "expected_error_type" in expected:
        scores.append(
            metric(
                "expected_error",
                result.error_type == expected["expected_error_type"],
            )
        )
    return scores


def retrieval_scores(*, output, expected_output=None, **kwargs):
    result = ExperimentOutput.model_validate(output)
    retrieval = result.retrieval or {}
    expected = expected_output or {}
    scores = [
        metric(
            "execution_success",
            result.error_type is None and result.retrieval is not None,
        ),
        metric("elapsed_ms", result.elapsed_ms),
    ]
    for field in (
        "model_calls",
        "input_tokens",
        "output_tokens",
        "unreported_calls",
    ):
        if field in retrieval:
            scores.append(metric(field, retrieval[field]))
    if "relevance" not in expected:
        return scores
    grades = expected["relevance"]
    if not isinstance(grades, dict) or any(
        not isinstance(v, (int, float))
        or isinstance(v, bool)
        or not isfinite(v)
        or not 0 <= v <= 3
        for v in grades.values()
    ):
        raise ValueError(
            "relevance must map document IDs to grades between 0 and 3"
        )
    relevant = {key for key, grade in grades.items() if grade > 0}
    for stage, hits in retrieval.get("stages", {}).items():
        ids = list(dict.fromkeys(hit["evaluation_id"] for hit in hits))
        k = retrieval["k"]
        ids = ids[:k]
        if not relevant:
            scores.append(metric(f"{stage}.expected_empty", not ids))
            continue
        found = sum(key in relevant for key in ids)
        rr = next(
            (
                1 / (rank + 1)
                for rank, key in enumerate(ids)
                if key in relevant
            ),
            0.0,
        )
        dcg = sum(
            (2 ** grades.get(key, 0) - 1) / log2(rank + 2)
            for rank, key in enumerate(ids)
        )
        ideal = sum(
            (2**grade - 1) / log2(rank + 2)
            for rank, grade in enumerate(
                sorted(grades.values(), reverse=True)[:k]
            )
        )
        scores.extend(
            [
                metric(f"{stage}.recall@{k}", found / len(relevant)),
                metric(f"{stage}.precision@{k}", found / k),
                metric(f"{stage}.mrr@{k}", rr),
                metric(f"{stage}.ndcg@{k}", dcg / ideal),
            ]
        )
    return scores


def acceptance_evaluator(
    expected_items, thresholds, *, judge=False, budget=None, saved=False
):
    """Missing items, failed evaluators and missing scores fail closed."""

    initial_calls = budget.calls if budget is not None else 0

    def evaluate_run(*, item_results, **kwargs):
        complete = len(item_results) == expected_items and expected_items > 0
        accepted = complete
        covered = 0
        for result in item_results:
            scores = {e.name: e.value for e in result.evaluations}
            required = dict(thresholds)
            item = result.item
            reference = (
                item.get("expected_output")
                if isinstance(item, dict)
                else item.expected_output
            ) or {}
            if "expected_status" in reference:
                required.setdefault("expected_status", 1.0)
            elif "grounded_findings" in scores:
                required.setdefault("grounded_findings", 1.0)
                if "required_source_citation" in scores:
                    required.setdefault("required_source_citation", 1.0)
            if judge and any(
                key in reference
                for key in ("reference_answer", "root_cause_service")
            ) and scores.get("diagnosis_status") == "complete":
                required.setdefault("root_cause_correctness", 1.0)
            supplied = item.get("input") if isinstance(
                item, dict) else item.input
            if saved and (supplied.get("run_state") or supplied.get("state")):
                required.setdefault("trace_integrity", 1.0)
                required.setdefault("tool_budget", 1.0)
            if "runtime_verification" in scores:
                required.setdefault("runtime_verification", 1.0)
            present = all(name in scores for name in required)
            covered += present
            accepted = (
                accepted
                and present
                and all(
                    isinstance(scores[name], (int, float))
                    and scores[name] >= minimum
                    for name, minimum in required.items()
                )
            )
        results = [
            metric(
                "evaluation_coverage",
                covered / expected_items if expected_items else 0,
            ),
            metric("acceptance", bool(accepted)),
        ]
        if budget is not None:
            results.extend(
                [
                    metric(
                        "model_calls_admitted", budget.calls - initial_calls
                    ),
                    metric("model_call_cap", budget.limit),
                    metric(
                        "model_calls_remaining", budget.limit - budget.calls
                    ),
                ]
            )
        return results

    return evaluate_run


def verification_scores(*, output, expected_output=None, **kwargs):
    """Measure review coverage and false acceptance on held-out subjects."""
    result = ExperimentOutput.model_validate(output)
    record = result.run_state.verification if result.run_state else None
    review = record.review if record and record.status == "completed" else None
    expected = (expected_output or {}).get("verdict")
    scores = [metric("execution_success", result.error_type is None),
              metric("review_completed", review is not None),
              metric("trace_integrity", bool(result.run_state
                     and not trace_issues(result.run_state))),
              metric("tool_budget", bool(result.run_state
                     and sum(result.run_state.tool_calls.values())
                     <= result.run_state.limits.tool_calls))]
    if result.run_state is not None:
        usage = result.run_state.usage
        scores.extend([
            metric("duration_ms", result.elapsed_ms),
            metric("model_calls", result.run_state.model_calls),
            metric("reported_input_tokens", usage.reported_input_tokens),
            metric("reported_output_tokens", usage.reported_output_tokens),
        ])
        if usage.reported_reasoning_model_calls:
            scores.append(metric("reported_reasoning_tokens",
                                 usage.reported_reasoning_tokens))
    if review is not None:
        scores.extend([
            metric("support_verdict", review.verdict),
            metric("would_accept", review.accepted),
        ])
    if expected is not None:
        scores.extend([
            metric(
                "expected_verdict",
                review is not None and review.verdict == expected),
            metric("expected_decision", review is not None
                   and review.accepted == (expected == "supported")),
            metric("false_acceptance", bool(review and review.accepted
                                            and expected != "supported")),
            metric("false_rejection", bool(review and not review.accepted
                                           and expected == "supported")),
        ])
    return scores
