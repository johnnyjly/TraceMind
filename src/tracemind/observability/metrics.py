"""Logical-run scores, independent of resume spans and model self-ratings."""

import logging
from uuid import NAMESPACE_URL, uuid5

PRICE_VERSION = "deepseek-peak-list-usd-2026-11-12"
PRICE_SOURCE = "https://api-docs.deepseek.com/quick_start/pricing/"
# USD per million tokens: cache hit, cache miss, output INCLUDING reasoning.
# Explicit peak-list estimate; not a time/holiday-aware invoice calculation.
PRICES = {
    "deepseek-flash": (0.006, 0.30, 1.20),
    "deepseek-v4-flash": (0.006, 0.30, 1.20),
    "deepseek-v4-pro": (0.044, 1.32, 3.96),
}
logger = logging.getLogger(__name__)


def estimate_cost(events):
    started = sum(e.kind == "model.started" for e in events)
    known, cost = 0, 0.0
    for event in events:
        if event.kind != "model.completed":
            continue
        data = event.data
        rates = PRICES.get(data.get("model"))
        counts = [data.get(k) for k in (
            "cache_hit_tokens", "cache_miss_tokens", "output_tokens",
        )]
        if (rates is None or not data.get("usage_reported")
                or not data.get("cache_usage_reported")
                or not all(type(n) is int and n >= 0 for n in counts)
                or sum(counts[:2]) != data.get("input_tokens")):
            continue
        known += 1
        cost += sum(n * rate / 1_000_000
                    for n, rate in zip(counts, rates))
    return {
        "priced_calls": known,
        "unpriced_calls": max(0, started - known),
        "cost_coverage": known / started if started else None,
        "known_cost_estimate_usd": cost if known else None,
        "total_cost_estimate_usd": (cost if known == started else None),
    }


def phase_duration(events, start_kind, end_kinds, finished_at):
    total, start, seen = 0.0, None, False
    for event in events:
        if event.kind == start_kind:
            start, seen = event.timestamp, True
        elif event.kind in end_kinds and start is not None:
            total += max(0, (event.timestamp - start).total_seconds() * 1000)
            start = None
    if start is not None and finished_at:
        total += max(0, (finished_at - start).total_seconds() * 1000)
    return total if seen else None


def run_metrics(state):
    if state.finished_at is None:
        raise ValueError("only terminal runs have final metrics")
    first = next((e.timestamp for e in state.events
                  if e.kind in {"triage.available", "diagnosis.delivered"}), None)
    records = [*state.verification_history,
               *([state.verification] if state.verification else [])]
    failure_reasons = {
        "deadline", "model_timeout", "tool_timeout", "model_error",
        "tool_error", "internal_error", "verification_failed",
    }
    reason = state.stop_reason.value if state.stop_reason else "unknown"
    scores = {
        "run_count": 1,
        "final_ms": max(0, (state.finished_at
                            - state.started_at).total_seconds() * 1000),
        "first_triage_ms": (max(0, (first - state.started_at).total_seconds()
                                * 1000) if first else None),
        "has_triage": int(first is not None),
        "execution_failed": int(state.status == "failed"
                                or reason in failure_reasons),
        "cancelled": int(state.status == "cancelled"),
        "deadline": int(reason == "deadline"),
        "context_limit": int(reason == "context_limit" or any(
            r.error_code == "context_limit" for r in records)),
        "review_completed": (int(records[-1].status == "completed"
                                 and records[-1].review is not None)
                             if records else None),
        "first_review_rejected": (
            int(not records[0].review.accepted)
            if records and records[0].status == "completed"
            and records[0].review else None),
        "final_review_rejected": (
            int(not records[-1].review.accepted)
            if records and records[-1].status == "completed"
            and records[-1].review else None),
        "review_rounds": len(records),
        "verification_ms": phase_duration(
            state.events, "verification.started",
            {"verification.completed", "verification.failed",
             "verification.cancelled"}, state.finished_at),
        "revision_ms": phase_duration(
            state.events, "investigation.revision_started",
            {"verification.started"}, state.finished_at),
        "reported_tokens": state.usage.reported_total_tokens,
        "input_tokens": state.usage.reported_input_tokens,
        "output_tokens": state.usage.reported_output_tokens,
        "reasoning_tokens": (state.usage.reported_reasoning_tokens
                             if state.usage.reported_reasoning_model_calls
                             else None),
        "cache_hit_tokens": (state.usage.reported_cache_hit_tokens
                             if state.usage.reported_cache_model_calls
                             else None),
        "usage_coverage": (state.usage.reported_model_calls / state.model_calls
                           if state.model_calls else None),
        **estimate_cost(state.events),
    }
    metadata = {
        **state.observability_metadata,
        "run_kind": state.observability_metadata.get("run_kind", "investigation"),
        "run_id": str(state.id), "incident_id": str(state.incident.id),
        "parent_run_id": str(state.parent_run_id) if state.parent_run_id else None,
        "status": state.status.value, "stop_reason": reason,
        "agent_mode": state.agent_mode,
        "prompt_versions": {role: {
            "name": p.name, "version": p.version, "sha256": p.sha256,
        } for role, p in state.prompts.items()},
        "price_version": PRICE_VERSION, "price_source": PRICE_SOURCE,
        "cost_basis": "peak_list_estimate_not_invoice",
        "time_basis": "wall_clock_including_waits_and_host_interruptions",
    }
    profile = state.observability_metadata.get("model_profile", {})
    for key in ("investigator_model", "reasoning_effort", "thinking"):
        if key in profile:
            metadata[key] = profile[key]
    if state.verification_policy:
        metadata["verifier_model"] = state.verification_policy.model
        metadata["verifier_reasoning_effort"] = (
            state.verification_policy.deepseek_reasoning_effort
        )
    for role, snapshot in state.prompts.items():
        metadata[role + "_prompt_version"] = snapshot.version
    return {"scores": scores, "metadata": metadata}


def publish_run_metrics(client, state):
    if client is None or state.finished_at is None:
        return
    try:
        summary = run_metrics(state)
        for key, value in summary["scores"].items():
            if value is None:
                continue
            name = "tracemind." + key
            client.create_score(
                score_id=str(uuid5(NAMESPACE_URL, str(state.id) + ":" + name)),
                session_id=str(state.id), name=name, value=value,
                data_type="NUMERIC", timestamp=state.finished_at,
                metadata=summary["metadata"],
            )
    except Exception as exc:
        # Observability must not replace a persisted incident result.
        logger.warning("Run metrics export failed (%s), run=%s",
                       type(exc).__name__, state.id)


def publish_review_score(client, *, trace_id, observation_id, record):
    if client is None or record.review is None:
        return
    try:
        client.create_score(
            score_id=str(uuid5(NAMESPACE_URL, trace_id + ":" + observation_id
                              + ":verdict")),
            trace_id=trace_id, observation_id=observation_id,
            name="tracemind.verifier.verdict", value=record.review.verdict,
            data_type="CATEGORICAL",
            metadata={"candidate_sha256": record.input_sha256,
                      "rubric_version": record.rubric_version},
        )
    except Exception as exc:
        logger.warning("Reviewer score export failed (%s)", type(exc).__name__)
