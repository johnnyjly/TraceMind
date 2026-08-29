"""Pure follow-up policy.

Handles the window, retained evidence, history and permissions.
"""
from datetime import datetime
from uuid import UUID

from tracemind.models import Evidence
from tracemind.models.memory import MemoryPolicy
from tracemind.models.run import (
    RunLimits,
    RunPurpose,
    RunState,
    RunStatus,
    ToolName,
)
from tracemind.models.storage import RunConflict, StoredRun
from tracemind.agent.workflow.tool_policy import resolve_tool_policy


def plan_follow_up(
    parent: StoredRun, history: list[RunState], *, investigation_id: UUID,
    message: str, purpose: RunPurpose, limits: RunLimits,
    allowed_tools: tuple[ToolName, ...], start: datetime | None,
    end: datetime | None, memory_policy: MemoryPolicy | None,
) -> tuple[RunState, list[Evidence]]:
    start = start or parent.state.investigation_start
    end = end or parent.state.investigation_end
    if (start.utcoffset() is None or end.utcoffset() is None
            or end <= start):
        raise ValueError("invalid follow-up investigation window")
    if purpose is RunPurpose.EXPLAIN and (
        start != parent.state.investigation_start or
        end != parent.state.investigation_end
    ):
        raise RunConflict("explanation must use the previous window")
    prior = parent.state.diagnosis or parent.state.prior_diagnosis
    candidates = [item for item in parent.evidence
                  if item.observed_at.utcoffset() is not None
                  and start <= item.observed_at <= end]
    cited = set(prior.evidence_ids) if prior else set()
    candidates.sort(key=lambda item: item.id not in cited)
    evidence = []
    size = 0
    for item in candidates:
        item_size = len(item.model_dump_json().encode())
        if size + item_size <= limits.observation_bytes:
            evidence.append(item)
            size += item_size
    state = RunState(
        incident=parent.state.incident,
        investigation_id=investigation_id,
        parent_run_id=parent.state.id, purpose=purpose,
        prompt_version="single-agent-followup-v3",
        user_message=message,
        previous_messages=[s.user_message for s in history
                           if s.user_message],
        prior_diagnosis=prior,
        omitted_prior_evidence=len(parent.evidence) - len(evidence),
        memory_policy=(memory_policy if purpose is
                       RunPurpose.INVESTIGATE else None),
        learning_policy=memory_policy,
        investigation_start=start, investigation_end=end,
        limits=limits.model_copy(deep=True),
        allowed_tools=resolve_tool_policy(allowed_tools, purpose=purpose),
        status=RunStatus.QUEUED,
    )
    return state, evidence
