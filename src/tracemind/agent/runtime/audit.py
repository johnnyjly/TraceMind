"""Audit retained execution events independently of evaluation tooling."""

from tracemind.models.run import RunState


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
