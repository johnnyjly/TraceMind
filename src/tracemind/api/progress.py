"""Project durable run events into readable, reconnectable SSE progress."""

import asyncio
from collections.abc import AsyncIterator
import json
from time import monotonic
from uuid import UUID

from fastapi import Request

from tracemind.agent.storage.run_repository import RunRepository
from tracemind.api.schemas import ProgressEvent
from tracemind.models.run import RunEvent
from tracemind.models.storage import TERMINAL


TOOLS = {
    "query_metrics": "monitoring metrics",
    "search_traces": "distributed traces",
    "get_trace_detail": "trace details", "search_logs": "service logs",
    "read_evidence": "saved evidence",
    "discover_metrics": "metric names and labels",
    "search_knowledge": "reference knowledge",
    "read_knowledge": "saved reference sections",
}
MESSAGES = {
    "collaboration.planned": ("planning", "Investigation tasks assigned"),
    "collaboration.handoff": (
        "synthesizing", "Checking investigator findings"),
    "collaboration.reinvestigation": (
        "investigating", "Investigating specific verification gaps"),
    "collaboration.task_incomplete": (
        "synthesizing", "Task budget reached; retaining evidence and gaps"),
    "verification.started": (
        "verifying", "Reviewing diagnosis against retained evidence"),
    "verification.completed": ("verified", "Diagnosis review completed"),
    "verification.failed": (
        "error", "Diagnosis review failed; completion blocked"),
    "verification.cancelled": ("cancelling", "Diagnosis review cancelled"),
    "run.queued": ("queued", "Investigation received, awaiting execution"),
    "run.started": ("started", "Starting investigation"),
    "run.resumed": ("resuming", "Resuming interrupted investigation"),
    "model.started": ("analyzing", "Analyzing available information"),
    "model.failed": ("error", "Model call failed"),
    "model.cancelled": ("cancelling", "Model call cancelled"),
    "recovery.retry_scheduled": (
        "retrying", "Call temporarily failed, retrying"),
    "model.switched": ("retrying", "Switching to fallback model"),
    "recovery.ambiguous_call": (
        "resuming", "The outcome of the interrupted call is unconfirmed"),
    "diagnosis.repair_requested": ("analyzing", "Correcting diagnosis output"),
    "policy.finalizing": (
        "finalizing", "Tool investigation stopped, preparing conclusions "
        "from available evidence"),
    "memory.selected": (
        "historical_context", "Historical experience retrieved, "
        "checking current evidence"),
    "guidance.selected": (
        "historical_context",
        "User preferences and investigation skill loaded"),
    "memory.feedback_saved": (
        "memory_updated", "Feedback memory processing completed"),
    "knowledge.search.started": ("querying", "Searching reference knowledge"),
    "knowledge.search.completed": (
        "query_completed", "Reference knowledge saved"),
    "knowledge.search.failed": (
        "error", "Reference knowledge search unavailable"),
}


def readable_event(event: RunEvent) -> ProgressEvent | None:
    """Allowlist display fields; prompts, raw queries and logs stay private."""
    data = {key: value for key, value in event.data.items() if key in {
        "tool", "service", "investigation_start", "investigation_end",
        "result_count", "reason", "status", "error_type", "category",
        "added", "received", "duplicates", "record_count",
        "applied", "skipped", "rejected_count", "verdict", "accepted", "code",
        "role", "phase", "task_id", "round",
    } and isinstance(value, (str, int, float, bool))}
    if event.kind.startswith("tool."):
        tool = TOOLS.get(data.get("tool"), "tool")
        service = data.get("service")
        target = f" {service}" if service else ""
        phase = "querying"
        if event.kind == "tool.started":
            message = (f"Reading {tool} for{target}"
                       if target else f"Reading {tool}")
        elif event.kind == "tool.completed":
            phase = "query_completed"
            count = data.get("result_count", 0)
            # result_count counts returned preview records, not new evidence.
            message = f"Finished reading {tool}, returned {count} records"
        elif event.kind == "tool.reused":
            phase, message = "query_completed", f"Reusing saved {tool}"
        elif event.kind == "tool.failed":
            phase, message = "error", f"Failed to read {tool}"
        elif event.kind == "tool.cancelled":
            phase, message = "cancelling", f"Reading {tool} cancelled"
        elif event.kind == "tool.skipped":
            phase, message = (
                "finalizing", f"Skipped {tool}, investigation is finalizing")
        else:
            return None
    elif event.kind == "observations.stored":
        phase = "evidence_received"
        message = f"Saved {data.get('added', 0)} new evidence records"
    elif event.kind == "run.finished":
        phase = "finished"
        message = {
            "succeeded": "Diagnosis completed",
            "inconclusive": (
                "Investigation finished without a confirmed root cause"),
            "cancelled": "Investigation cancelled",
            "failed": "Investigation failed",
        }.get(data.get("status"), "Investigation finished")
    elif event.kind in {"agent.phase_started", "agent.phase_completed"}:
        phase = str(data.get("phase", "analyzing"))
        action = ("Starting" if event.kind.endswith("started") else "Finished")
        message = f"{action} {data.get('role', 'agent')} {phase}"
    elif event.kind in MESSAGES:
        phase, message = MESSAGES[event.kind]
    else:
        return None
    return ProgressEvent(
        run_id=event.run_id, sequence=event.sequence,
        span_id=event.span_id, parent_span_id=event.parent_span_id,
        timestamp=event.timestamp, kind=event.kind,
        phase=phase, message=message, data=data,
    )


async def stream_progress(
    repository: RunRepository, run_id: UUID, request: Request, *,
    after: int = 0, poll_seconds: float = .5, heartbeat_seconds: float = 15,
) -> AsyncIterator[str]:
    """Replay progress then tail SQLite; disconnect affects only this reader.

    Durable sequence numbers are cursors. Delivery is at least once across
    reconnects, so clients deduplicate by run ID and sequence. Completion is
    emitted only after the terminal row/result transaction commits.
    """
    cursor = after
    heartbeat = monotonic()
    while not await request.is_disconnected():
        snapshot = await asyncio.to_thread(repository.event_snapshot, run_id)
        for event in snapshot.events:
            if event.sequence <= cursor:
                continue
            cursor = event.sequence
            progress = readable_event(event)
            if progress is not None:
                yield (f"id: {event.sequence}\nevent: progress\n"
                       f"data: {progress.model_dump_json()}\n\n")
        if snapshot.status in TERMINAL:
            result = (f"/api/v1/runs/{run_id}/result"
                      if snapshot.has_result else None)
            yield (f"id: {cursor}\nevent: complete\ndata: " + json.dumps({
                "run_id": str(run_id), "status": snapshot.status.value,
                "result_url": result,
            }) + "\n\n")
            return
        if monotonic() - heartbeat >= heartbeat_seconds:
            yield ": keep-alive\n\n"
            heartbeat = monotonic()
        await asyncio.sleep(poll_seconds)
