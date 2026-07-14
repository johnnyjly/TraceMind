import asyncio
from datetime import UTC, datetime

from httpx import ASGITransport, AsyncClient

from tracemind.api.app import app, get_diagnosis_service, get_request_settings
from tracemind.config import Settings
from tracemind.models import Diagnosis, DiagnosisStatus


class StubDiagnosisService:
    async def diagnose(self, incident, *, start, end):
        assert start == datetime(2024, 9, 1, tzinfo=UTC)
        assert end == datetime(2024, 9, 1, 1, tzinfo=UTC)
        return Diagnosis(
            incident_id=incident.id,
            status=DiagnosisStatus.INCONCLUSIVE,
            root_cause="Not enough telemetry",
            confidence=0.2,
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
