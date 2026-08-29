"""Restart, lease, cancellation, and HTTP contracts for persistent runs."""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3

from httpx import ASGITransport, AsyncClient
import pytest

from tracemind.agent.storage.run_repository import RunRepository
from tracemind.api.app import (
    app,
    get_agent_runner,
    get_request_settings,
    get_run_repository,
)
from tracemind.config import Settings
from tracemind.models import Evidence, EvidenceSource, Incident
from tracemind.models.run import RunLimits, RunState, RunStatus, StopReason
from tracemind.models.storage import RunConflict


START = datetime(2024, 9, 1, tzinfo=UTC)


def queued_state() -> RunState:
    """Build a queued run with fixed incident and investigation window."""
    return RunState(
        incident=Incident(title="checkout errors", description="HTTP 500"),
        investigation_start=START,
        investigation_end=START + timedelta(hours=1),
        limits=RunLimits(), allowed_tools=("search_traces",),
        status=RunStatus.QUEUED,
    )


def expire_lease(path: Path, run_id) -> None:
    """Simulate a process dying without its heartbeat."""
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE runs SET lease_until=0 WHERE id=?", (str(run_id),)
        )


def test_restart_retains_evidence_and_budgets(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite3"
    repository = RunRepository(path)
    state = queued_state()
    repository.create(state, request_hash="request")
    claim = repository.claim_next()
    assert claim is not None
    assert repository.claim_next() is None
    claim.run.state.model_calls = 2
    claim.run.state.tool_calls = {"search_traces": 1}
    claim.run.state.tool_signatures = {"digest": 1}
    item = Evidence(
        source=EvidenceSource.TRACE, summary="payment failed",
        observed_at=START, service="payment",
    )
    repository.save_progress(
        state.id, claim.owner, claim.run.state, [item]
    )

    restarted = RunRepository(path)
    expire_lease(path, state.id)
    resumed = restarted.claim_next()
    assert resumed is not None and resumed.resuming
    assert resumed.run.state.model_calls == 2
    assert resumed.run.state.tool_calls == {"search_traces": 1}
    assert resumed.run.state.tool_signatures == {"digest": 1}
    assert [evidence.id for evidence in resumed.run.evidence] == [item.id]
    with pytest.raises(RunConflict, match="lease was lost"):
        repository.save_progress(state.id, claim.owner, claim.run.state, [])


def test_idempotency_resume_and_cancellation(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite3"
    repository = RunRepository(path)
    state = queued_state()
    first = repository.create(
        state, request_hash="same", idempotency_key="request-1"
    )
    repeated = repository.create(
        queued_state(), request_hash="same", idempotency_key="request-1"
    )
    assert repeated.state.id == first.state.id
    with pytest.raises(RunConflict, match="different request"):
        repository.create(
            queued_state(), request_hash="other",
            idempotency_key="request-1",
        )
    claim = repository.claim_next()
    assert claim is not None
    with pytest.raises(RunConflict, match="active worker"):
        repository.resume(state.id)
    repository.cancel(state.id)
    assert repository.should_stop(state.id, claim.owner)
    expire_lease(path, state.id)
    cancelled = repository.cancel(state.id)
    assert cancelled.state.status is RunStatus.CANCELLED
    assert cancelled.state.stop_reason is StopReason.CANCELLED
    assert repository.claim_next() is None


def test_expired_lease_can_be_requeued_once(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite3"
    repository = RunRepository(path)
    state = queued_state()
    repository.create(state, request_hash="request")
    first = repository.claim_next()
    assert first is not None
    expire_lease(path, state.id)
    assert repository.resume(state.id).state.status is RunStatus.QUEUED
    second = repository.claim_next()
    assert second is not None and second.resuming
    assert second.owner != first.owner
    assert repository.claim_next() is None


def test_run_api_queues_reads_and_cancels(tmp_path: Path) -> None:
    repository = RunRepository(tmp_path / "runs.sqlite3")

    async def override_repository():
        return repository

    async def override_runner():
        return object()

    async def override_settings():
        return Settings(_env_file=None)

    async def requests():
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            payload = {
                "incident": {"title": "checkout", "description": "HTTP 500"},
                "investigation_start": "2024-09-01T00:00:00Z",
                "investigation_end": "2024-09-01T01:00:00Z",
            }
            headers = {"Idempotency-Key": "api-request-1"}
            created = await client.post(
                "/api/v1/runs", json=payload, headers=headers
            )
            repeated = await client.post(
                "/api/v1/runs", json=payload, headers=headers
            )
            changed = await client.post(
                "/api/v1/runs",
                json={**payload, "incident": {
                    **payload["incident"],
                    "id": "1a2e78ab-1c0f-487d-a589-a59497bd48d3",
                }},
                headers=headers,
            )
            run_id = created.json()["id"]
            status = await client.get(f"/api/v1/runs/{run_id}")
            pending = await client.get(f"/api/v1/runs/{run_id}/result")
            cancelled = await client.post(f"/api/v1/runs/{run_id}/cancel")
            return created, repeated, changed, status, pending, cancelled

    app.dependency_overrides[get_run_repository] = override_repository
    app.dependency_overrides[get_agent_runner] = override_runner
    app.dependency_overrides[get_request_settings] = override_settings
    try:
        created, repeated, changed, status, pending, cancelled = (
            asyncio.run(requests())
        )
    finally:
        app.dependency_overrides.clear()

    assert created.status_code == 202
    assert created.headers["location"].endswith(created.json()["id"])
    assert created.json()["id"] == repeated.json()["id"]
    assert changed.status_code == 409
    assert status.json()["status"] == "queued"
    assert pending.status_code == 409
    assert cancelled.json()["status"] == "cancelled"
