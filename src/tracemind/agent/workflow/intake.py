"""Bounded lookup of supplied trace locators before model investigation."""

from tracemind.agent.runtime.harness import (
    IntakeBudgetExhausted, InvestigationClosed,
)
from tracemind.agent.runtime.recovery import ToolUnavailable
from tracemind.models import EvidenceSource
from tracemind.models.run import IntakeProgress, IntakeResult, RunPurpose


def _matching_evidence(target, evidence):
    return [item for item in evidence if (
        item.source is EvidenceSource.TRACE
        and isinstance(item.attributes.get("trace_id"), str)
        and item.attributes["trace_id"].lower() == target.trace_id
        and isinstance(item.attributes.get("span_id"), str)
        and (target.span_id is None
             or item.attributes["span_id"].lower() == target.span_id)
        and item.attributes.get("span_id")
    )]


def _result(index, target, evidence, output):
    matches = _matching_evidence(target, evidence)
    coverage = next((item.attributes["coverage"] for item in matches
                     if isinstance(item.attributes.get("coverage"), dict)),
                    next((item for item in output
                          if item.get("kind") == "trace_detail_coverage"), {}))
    coverage = {key: value for key, value in coverage.items()
                if key in {"trace_span_count", "matched_span_count",
                           "page_offset", "page_size", "next_offset"}
                and (value is None or type(value) is int and value >= 0)}
    if matches:
        confirmed_hint = any(
            (target.service is None or item.service == target.service)
            and (target.operation is None
                 or item.attributes.get("operation") == target.operation)
            for item in matches
        )
        return IntakeResult(
            target_index=index, status="found",
            evidence_ids=[item.id for item in matches],
            reason=(None if confirmed_hint
                    else "reported_locator_hints_unconfirmed"),
            backend_coverage=coverage,
        )
    for field, status in (
        ("excluded_scope_mismatch", "scope_mismatch"),
        ("excluded_scope_unknown", "scope_unknown"),
        ("excluded_outside_window", "outside_window"),
    ):
        if any(item.get(field) for item in output):
            return IntakeResult(target_index=index, status=status,
                                backend_coverage=coverage)
    return IntakeResult(
        target_index=index, status="coverage_incomplete",
        reason="no_matching_retained_span",
        backend_coverage=coverage,
    )


async def collect_initial_evidence(state, collector, harness, tools):
    """Use the same authorized tools, admission ledger and evidence store."""
    context = state.initial_context
    if context is None or state.purpose is RunPurpose.EXPLAIN:
        return
    if state.intake_progress is None:
        state.intake_progress = IntakeProgress()
    progress = state.intake_progress
    if progress.completed:
        return
    tool = next((t for t in tools if t.name == "get_trace_detail"), None)
    completed = {result.target_index for result in progress.results}
    harness.emit("intake.started", data={"target_count": len(context.traces)})
    await harness.checkpoint()
    harness.intake_active = True
    try:
        for index, target in enumerate(context.traces):
            if index in completed:
                continue
            harness.check_active()
            result = None
            matches = _matching_evidence(target, collector.evidence)
            if not context.prefetch_traces:
                result = IntakeResult(
                    target_index=index, status="skipped",
                    reason="prefetch_disabled",
                )
            elif matches:
                result = _result(index, target, matches, [])
                result.reason = result.reason or "retained_observation"
            elif tool is None:
                result = IntakeResult(
                    target_index=index, status="skipped",
                    reason="tool_not_authorized",
                )
            elif progress.attempts >= progress.attempt_limit:
                result = IntakeResult(
                    target_index=index, status="skipped",
                    reason="intake_attempt_budget",
                )
            else:
                # Service/operation are reported hints, not identity filters.
                arguments = {"trace_id": target.trace_id}
                if target.span_id is not None:
                    arguments["span_id"] = target.span_id
                properties = tool.args
                if target.span_id and "span_id" not in properties:
                    result = IntakeResult(
                        target_index=index, status="skipped",
                        reason="unsupported_span_filter",
                    )
                else:
                    try:
                        output = await tool.ainvoke(arguments)
                        result = _result(index, target, collector.evidence,
                                         output)
                    except ToolUnavailable as exc:
                        result = IntakeResult(
                            target_index=index,
                            status=("not_found" if exc.failure.category
                                    == "not_found" else "unavailable"),
                            reason=exc.failure.category,
                        )
                    except IntakeBudgetExhausted:
                        result = IntakeResult(
                            target_index=index, status="skipped",
                            reason="intake_attempt_budget",
                        )
                    except InvestigationClosed:
                        result = IntakeResult(
                            target_index=index, status="skipped",
                            reason="investigation_tools_closed",
                        )
            progress.results.append(result)
            harness.emit("intake.target_completed", data={
                "target_index": index, "status": result.status,
                "reason": result.reason,
                "result_count": len(result.evidence_ids),
            })
            await harness.checkpoint()
        progress.completed = True
        harness.emit("intake.completed", data={
            "target_count": len(progress.results),
            "attempts": progress.attempts,
        })
        await harness.checkpoint()
    finally:
        harness.intake_active = False
