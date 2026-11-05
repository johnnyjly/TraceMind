"""Exercise context admission, persistence and follow-up without models."""

import asyncio
import sqlite3
from uuid import UUID

from httpx import ASGITransport, AsyncClient
import pytest

from tracemind.agent.storage.run_repository import RunRepository
from tracemind.api.app import (
    app, get_agent_runner, get_diagnosis_service, get_request_settings,
    get_run_repository,
)
from tracemind.config import Settings
from tracemind.models import Diagnosis, DiagnosisStatus
from tracemind.models.intake import InitialContext
from tracemind.models.result import DiagnosisRun


PAYLOAD = {
    "incident": {"title": "checkout", "description": "HTTP 500"},
    "investigation_start": "2024-09-01T00:00:00Z",
    "investigation_end": "2024-09-01T01:00:00Z",
}
CONTEXT = {
    "traces": [{"trace_id": "AB" * 16, "span_id": "CD" * 8}],
    "error": {"message": "upstream failed; password=never-persist-this"},
    "scope": {"environment": "production"},
    "deployments": [{"service": "checkout", "version": "v2"}],
}


def test_sync_endpoint_passes_sanitized_typed_context():
    captured = []

    class Service:
        async def diagnose_with_evidence(self, incident, *, start, end,
                                         initial_context):
            captured.append(initial_context)
            return DiagnosisRun(
                diagnosis=Diagnosis(incident_id=incident.id,
                                    status=DiagnosisStatus.INCONCLUSIVE,
                                    root_cause="Need evidence", confidence=0),
                evidence=[], tool_calls={},
            )

    async def request():
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as client:
            return await client.post("/api/v1/diagnoses", json={
                **PAYLOAD, "initial_context": CONTEXT,
            })

    app.dependency_overrides[get_diagnosis_service] = lambda: Service()
    try:
        response = asyncio.run(request())
    finally:
        app.dependency_overrides.pop(get_diagnosis_service)
    assert response.status_code == 200
    assert len(captured) == 1
    assert isinstance(captured[0], InitialContext)
    assert captured[0].traces[0].trace_id == "ab" * 16
    assert "never-persist-this" not in captured[0].model_dump_json()


def test_queue_and_followup_keep_only_current_round_context(tmp_path):
    repository = RunRepository(tmp_path / "intake.sqlite3")
    configured = Settings(_env_file=None)

    async def requests():
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as client:
            for url in ("/api/v1/runs", "/api/v1/investigations"):
                body = {**PAYLOAD, "initial_context": CONTEXT}
                headers = {"Idempotency-Key": url}
                created = await client.post(url, json=body, headers=headers)
                repeated = await client.post(url, json=body, headers=headers)
                assert created.status_code == repeated.status_code == 202
                assert created.json() == repeated.json()
                result = created.json()
                run_id = (result["run"]["id"] if "run" in result
                          else result["id"])
                state = repository.get(UUID(run_id)).state
                assert state.initial_context == InitialContext(**CONTEXT)
                assert state.evidence_ids == []
                assert state.tool_calls == {}
                changed = {**CONTEXT, "traces": [{"trace_id": "e" * 32}]}
                conflict = await client.post(url, json={
                    **PAYLOAD, "initial_context": changed,
                }, headers=headers)
                assert conflict.status_code == 409
                if "run" not in result:
                    continue
                message_url = f"{url}/{result['id']}/messages"
                await client.post(f"/api/v1/runs/{run_id}/cancel")
                no_context = await client.post(message_url, json={
                    "message": "test another hypothesis",
                })
                assert no_context.status_code == 202
                next_id = no_context.json()["id"]
                next_state = repository.get(UUID(next_id)).state
                assert next_state.initial_context is None
                await client.post(f"/api/v1/runs/{next_id}/cancel")
                fresh = await client.post(message_url, json={
                    "message": "check this request",
                    "initial_context": changed,
                    "investigation_start": "2024-09-02T00:00:00Z",
                    "investigation_end": "2024-09-02T01:00:00Z",
                })
                assert fresh.status_code == 202
                fresh_state = repository.get(UUID(fresh.json()["id"])).state
                assert fresh_state.initial_context == InitialContext(**changed)
                assert fresh_state.investigation_start.day == 2
                assert repository.get(UUID(run_id)).state.initial_context == (
                    InitialContext(**CONTEXT)
                )
                await client.post(f"/api/v1/runs/{fresh_state.id}/cancel")
                rejected = await client.post(message_url, json={
                    "message": "explain", "purpose": "explain",
                    "initial_context": CONTEXT,
                })
                assert rejected.status_code == 422
                assert "never-persist-this" not in rejected.text
                explained = await client.post(message_url, json={
                    "message": "explain", "purpose": "explain",
                })
                assert explained.status_code == 202
                explanation = repository.get(UUID(explained.json()["id"]))
                assert explanation.state.initial_context is None
                assert ("get_trace_detail"
                        not in explanation.state.allowed_tools)

    overrides = {get_run_repository: lambda: repository,
                 get_request_settings: lambda: configured,
                 get_agent_runner: lambda: object()}
    app.dependency_overrides.update(overrides)
    try:
        asyncio.run(requests())
    finally:
        for dependency in overrides:
            app.dependency_overrides.pop(dependency)
    with sqlite3.connect(repository.path) as connection:
        snapshots = connection.execute(
            "SELECT state_json FROM runs"
        ).fetchall()
    assert all("never-persist-this" not in state for (state,) in snapshots)


@pytest.mark.parametrize("invalid_context", [
    {"error": {"message": "password=do-not-echo", "line": -1}},
    {"unknown": "do-not-echo"},
    {"error": {"stacktrace": "do-not-echo" * 4000}},
])
def test_rejected_context_does_not_echo_sensitive_input(
    tmp_path, invalid_context,
):
    repository = RunRepository(tmp_path / "rejected.sqlite3")

    async def request():
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as client:
            return await client.post("/api/v1/runs", json={
                **PAYLOAD, "initial_context": invalid_context,
            })

    overrides = {get_run_repository: lambda: repository,
                 get_request_settings: lambda: Settings(_env_file=None)}
    app.dependency_overrides.update(overrides)
    try:
        response = asyncio.run(request())
    finally:
        for dependency in overrides:
            app.dependency_overrides.pop(dependency)
    assert response.status_code == 422
    assert "do-not-echo" not in response.text
    assert all("input" not in error for error in response.json()["detail"])
    with sqlite3.connect(repository.path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM runs").fetchone()
        assert count == (0,)
