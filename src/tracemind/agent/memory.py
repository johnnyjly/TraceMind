"""Deterministic retrieval from an explicit, held-out memory snapshot."""

from collections.abc import Callable
from datetime import datetime

from tracemind.models import Incident
from tracemind.models.memory import MemoryPolicy, MemorySnapshot


MemoryLookup = Callable[[MemoryPolicy, Incident, datetime], MemorySnapshot]


def frozen_memory_lookup(snapshot: MemorySnapshot) -> MemoryLookup:
    """Copy a corpus so evaluation never consults a mutable live memory DB.

    The caller freezes it before held-out incidents. Ground truth and replay
    scripts are not inputs here. Expiry is evaluated at the snapshot's time.
    """
    frozen = snapshot.model_copy(deep=True)

    def lookup(
        policy: MemoryPolicy, incident: Incident, at: datetime,
    ) -> MemorySnapshot:
        """Select eligible history using the frozen reference time."""
        if policy.scope != frozen.scope:
            raise ValueError("memory snapshot scope does not match policy")
        records = frozen.eligible_records(
            scope=policy.scope, services=incident.affected_services,
            incident_id=incident.id, at=frozen.selected_at,
        )
        return MemorySnapshot(
            scope=frozen.scope, selected_at=frozen.selected_at,
            records=records[:policy.max_records],
        )

    return lookup


def evaluation_memory(
    snapshot: MemorySnapshot | None, policy: MemoryPolicy | None, *,
    start: datetime, max_records: int, context_bytes: int,
) -> tuple[MemoryPolicy | None, MemoryLookup | None]:
    """Opt in only through a corpus frozen before the evaluation window.

    Application memory enablement never selects the live library in evals.
    If configured, deployment scope must match the supplied corpus.
    """
    if snapshot is None:
        return None, None
    if snapshot.selected_at > start:
        raise ValueError(
            "memory corpus must be frozen before evaluation start")
    if policy is not None and policy.scope != snapshot.scope:
        raise ValueError("memory snapshot scope does not match configuration")
    selected_policy = MemoryPolicy(
        scope=snapshot.scope, max_records=max_records,
        context_bytes=context_bytes,
    )
    # Preserve the corpus timestamp, but reject expiry before the held-out
    # incident begins even if that record was valid when the corpus froze.
    eligible = snapshot.model_copy(deep=True, update={
        "records": [r for r in snapshot.records if r.expires_at > start],
    })
    return selected_policy, frozen_memory_lookup(eligible)
