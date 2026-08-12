import asyncio
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from tracemind.agent import DiagnosisError, DiagnosisRun
from tracemind.api.app import app, get_diagnosis_service, get_request_settings
from tracemind.config import Settings
from tracemind.models import Diagnosis, DiagnosisStatus, Incident
from tracemind.models.run import RunLimits, RunState, RunStatus, StopReason


class StubDiagnosisService:
    async def diagnose_with_evidence(self, incident, *, start, end):
        assert start == datetime(2024, 9, 1, tzinfo=UTC)
        assert end == datetime(2024, 9, 1, 1, tzinfo=UTC)
        return DiagnosisRun(
            diagnosis=Diagnosis(
                incident_id=incident.id,
                status=DiagnosisStatus.INCONCLUSIVE,
                root_cause="Not enough telemetry",
                confidence=0.2,
            ), evidence=[], tool_calls={},
            state=RunState(
                incident=incident,
                investigation_start=start, investigation_end=end,
                limits=RunLimits(), allowed_tools=(),
                status=RunStatus.INCONCLUSIVE,
                stop_reason=StopReason.INSUFFICIENT_EVIDENCE,
            ),
        )


def test_diagnosis_endpoint_returns_structured_result() -> None:
    async def override_service():
        return StubDiagnosisService()

    async def request_diagnosis():
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/api/v1/diagnoses",
                json={
                    "incident": {
                        "title": "checkout failure",
                        "description": "HTTP 500 spike",
                    },
                    "investigation_start": "2024-09-01T00:00:00Z",
                    "investigation_end": "2024-09-01T01:00:00Z",
                },
            )

    app.dependency_overrides[get_diagnosis_service] = override_service
    try:
        response = asyncio.run(request_diagnosis())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "inconclusive"
    assert response.json()["root_cause"] == "Not enough telemetry"
    assert response.headers["x-tracemind-run-id"]
    assert (
        response.headers["x-tracemind-stop-reason"] == "insufficient_evidence"
    )


def test_diagnosis_endpoint_rejects_reversed_window() -> None:
    async def override_service():
        return StubDiagnosisService()

    async def request_diagnosis():
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/api/v1/diagnoses",
                json={
                    "incident": {
                        "title": "checkout failure",
                        "description": "HTTP 500 spike",
                    },
                    "investigation_start": "2024-09-01T01:00:00Z",
                    "investigation_end": "2024-09-01T00:00:00Z",
                },
            )

    app.dependency_overrides[get_diagnosis_service] = override_service
    try:
        response = asyncio.run(request_diagnosis())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_diagnosis_endpoint_requires_selected_provider_key() -> None:
    async def override_settings():
        return Settings(_env_file=None, google_api_key=None)

    async def request_diagnosis():
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/api/v1/diagnoses",
                json={
                    "incident": {
                        "title": "checkout failure",
                        "description": "HTTP 500 spike",
                    },
                    "investigation_start": "2024-09-01T00:00:00Z",
                    "investigation_end": "2024-09-01T01:00:00Z",
                },
            )

    app.dependency_overrides[get_request_settings] = override_settings
    try:
        response = asyncio.run(request_diagnosis())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "TRACEMIND_GOOGLE_API_KEY" in response.json()["detail"]


@pytest.mark.parametrize("reason, expected_status", [
    (StopReason.DEADLINE, 504),
    (StopReason.TOOL_TIMEOUT, 504),
    (StopReason.INVALID_DIAGNOSIS, 502),
])
def test_api_exposes_safe_run_failure_metadata(reason, expected_status):
    state = RunState(
        incident=Incident(title="failure", description="test"),
        investigation_start=datetime(2024, 9, 1, tzinfo=UTC),
        investigation_end=datetime(2024, 9, 1, 1, tzinfo=UTC),
        limits=RunLimits(), allowed_tools=(),
        status=RunStatus.FAILED, stop_reason=reason,
    )

    class FailedService:
        async def diagnose_with_evidence(self, incident, *, start, end):
            raise DiagnosisError("Diagnosis failed", state=state)

    async def override_service():
        return FailedService()

    async def request():
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post("/api/v1/diagnoses", json={
                "incident": {"title": "failure", "description": "test"},
                "investigation_start": "2024-09-01T00:00:00Z",
                "investigation_end": "2024-09-01T01:00:00Z",
            })

    app.dependency_overrides[get_diagnosis_service] = override_service
    try:
        response = asyncio.run(request())
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == expected_status
    assert response.json() == {"detail": "Diagnosis failed"}
    assert response.headers["x-tracemind-run-id"] == str(state.id)
    assert response.headers["x-tracemind-stop-reason"] == reason.value


def test_diagnosis_request_requires_timezone():
    from tracemind.api.schemas import DiagnosisRequest

    with pytest.raises(ValueError, match="timezone"):
        DiagnosisRequest(
            incident=Incident(title="failure", description="test"),
            investigation_start="2024-09-01T00:00:00",
            investigation_end="2024-09-01T01:00:00",
        )
