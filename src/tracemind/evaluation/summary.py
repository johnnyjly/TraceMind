"""Aggregate evaluation outcomes and audit observable execution traces."""

from math import ceil
from statistics import mean

from pydantic import BaseModel, Field

from tracemind.evaluation.models import EvaluationReport
from tracemind.models import DiagnosisStatus
from tracemind.models.run import RunState, RunStatus


class TokenPricing(BaseModel):
    """Explicit USD rates per million tokens, with no provider defaults."""

    input_per_million: float = Field(ge=0, allow_inf_nan=False)
    output_per_million: float = Field(ge=0, allow_inf_nan=False)


def trace_issues(state: RunState) -> list[str]:
    """Check sequence, span closure, terminal outcome and call accounting."""
    issues = []
    events = state.events
    if [e.sequence for e in events] != list(range(1, len(events) + 1)):
        issues.append("event_sequence")
    if any(e.run_id != state.id for e in events):
        issues.append("run_isolation")
    terminal = [e for e in events if e.kind == "run.finished"]
    if len(terminal) != 1 or events[-1].kind != "run.finished":
        issues.append("terminal_event")
    elif terminal[0].data.get("status") != state.status.value:
        issues.append("terminal_status")
    if state.memory_snapshot is not None:
        selected = [e for e in events if e.kind == "memory.selected"]
        if len(selected) != 1 or selected[0].data.get("snapshot_id") != (
            state.memory_snapshot.fingerprint
        ):
            issues.append("memory_snapshot")
        elif any(e.kind == "model.started" and
                 e.sequence < selected[0].sequence for e in events):
            issues.append("memory_selection_order")
    for kind, expected in (("model", state.model_calls),
                           ("tool", sum(state.tool_calls.values()))):
        starts = [e.span_id for e in events if e.kind == f"{kind}.started"]
        ends = [e.span_id for e in events if e.kind in {
            f"{kind}.completed", f"{kind}.failed", f"{kind}.cancelled",
        }]
        ambiguous = {e.span_id for e in events
                     if e.kind == "recovery.ambiguous_call"}
        if len(starts) != expected or len(starts) != len(set(starts)):
            issues.append(f"{kind}_accounting")
        if (len(ends) != len(set(ends)) or not set(ends) <= set(starts)
                or set(starts) - set(ends) - ambiguous):
            issues.append(f"{kind}_span_closure")
    return issues


class EvaluationSummary(BaseModel):
    """Batch metrics with explicit missing trace, usage and pricing counts."""

    runs: int
    regression_passed: int
    passed: int
    pass_rate: float
    failed_runs: int
    cancelled_runs: int
    complete_diagnoses: int
    false_completions: int
    false_completion_rate: float | None
    model_calls: int
    tool_calls: int
    backend_attempts: int
    retries: int
    recovered_runs: int
    mean_elapsed_ms: float
    p95_elapsed_ms: float
    reported_tokens: int
    unreported_model_calls: int
    runs_without_state: int
    trace_valid_runs: int
    trace_issues: dict[str, list[str]]
    priced_runs: int
    estimated_cost_usd: float | None


class BatchReport(BaseModel):
    """Per-run reports and an aggregate, keeping execution modes explicit."""

    reports: list[EvaluationReport]
    summary: EvaluationSummary
    groups: dict[str, EvaluationSummary] = Field(default_factory=dict)


def build_batch_report(
    reports: list[EvaluationReport], *,
    pricing: dict[str, TokenPricing] | None = None,
) -> BatchReport:
    """Keep scripted and real model measurements in identifiable groups."""
    groups: dict[str, list[EvaluationReport]] = {}
    for report in reports:
        key = (f"{report.execution_mode}/"
               f"{report.model_provider}:{report.model_name}")
        if report.memory_mode == "frozen":
            key += f"/memory:{report.memory_corpus_id}"
        if report.telemetry_transport != "direct":
            key += f"/telemetry:{report.telemetry_transport}"
        groups.setdefault(key, []).append(report)
    return BatchReport(
        reports=reports, summary=summarize_reports(reports, pricing=pricing),
        groups={key: summarize_reports(items, pricing=pricing)
                for key, items in groups.items()},
    )


def summarize_reports(
    reports: list[EvaluationReport], *,
    pricing: dict[str, TokenPricing] | None = None,
) -> EvaluationSummary:
    """Compute heuristic scores; never turn missing usage into zero cost.

    Pricing keys are provider:model. Fallback runs require per-model usage
    attribution and are left unpriced by this initial aggregate.
    """
    states = [r.run_state for r in reports if r.run_state is not None]
    complete = sum(r.diagnosis is not None and
                   r.diagnosis.status is DiagnosisStatus.COMPLETE
                   for r in reports)
    false = sum(bool(r.result and r.result.false_completion) for r in reports)
    passed = sum(bool(r.result and r.result.passed) for r in reports)
    durations = sorted(r.elapsed_ms or
                       (r.result.elapsed_ms if r.result else 0)
                       for r in reports)
    audits = {str(s.id): trace_issues(s) for s in states}
    costs = []
    for report in reports:
        state = report.run_state
        rate = (pricing or {}).get(
            f"{report.model_provider}:{report.model_name}"
        )
        if (state is not None and rate is not None
                and not state.model_switches
                and state.usage.unreported_model_calls == 0
                and state.usage.reported_model_calls == state.model_calls):
            costs.append((state.usage.reported_input_tokens
                          * rate.input_per_million
                          + state.usage.reported_output_tokens
                          * rate.output_per_million) / 1_000_000)
    return EvaluationSummary(
        runs=len(reports), passed=passed,
        regression_passed=sum(r.regression_passed is True for r in reports),
        pass_rate=passed / len(reports) if reports else 0,
        failed_runs=sum(bool(r.error_type) and
                        r.error_type != "CancelledError" and
                        (r.run_state is None or
                         r.run_state.status is not RunStatus.CANCELLED)
                        for r in reports),
        cancelled_runs=sum(
            r.error_type == "CancelledError" or
            (r.run_state is not None and
             r.run_state.status is RunStatus.CANCELLED) for r in reports
        ),
        complete_diagnoses=complete, false_completions=false,
        false_completion_rate=false / complete if complete else None,
        model_calls=sum(s.model_calls for s in states),
        tool_calls=sum(sum(s.tool_calls.values()) for s in states),
        backend_attempts=sum(s.backend_attempts for s in states),
        retries=sum(e.kind == "recovery.retry_scheduled"
                    for s in states for e in s.events),
        recovered_runs=sum(any(e.kind == "recovery.retry_scheduled"
                               for e in s.events) and
                           s.status is RunStatus.SUCCEEDED for s in states),
        mean_elapsed_ms=mean(durations) if durations else 0,
        p95_elapsed_ms=durations[ceil(len(durations) * .95) - 1]
        if durations else 0,
        reported_tokens=sum(s.usage.reported_total_tokens for s in states),
        unreported_model_calls=sum(s.usage.unreported_model_calls
                                   for s in states),
        runs_without_state=len(reports) - len(states),
        trace_valid_runs=sum(not issues for issues in audits.values()),
        trace_issues={key: value for key, value in audits.items() if value},
        priced_runs=len(costs),
        estimated_cost_usd=sum(costs) if len(costs) == len(reports)
        and reports else None,
    )
