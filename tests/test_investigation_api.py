"""Validate incident collaboration APIs without invoking external models."""

import asyncio
import sqlite3
from uuid import uuid4

from httpx import ASGITransport, AsyncClient

from tracemind.agent.persistence import RunRepository
from tracemind.api.app import (
    app, get_agent_runner, get_request_settings, get_run_repository,
)
from tracemind.config import Settings


PAYLOAD = {
    "incident": {"title": "checkout", "description": "HTTP 500"},
    "investigation_start": "2024-09-01T00:00:00Z",
    "investigation_end": "2024-09-01T01:00:00Z",
}


def test_investigation_creation_followup_history_and_idempotency(tmp_path):
    repository = RunRepository(tmp_path / "runs.sqlite3")

    async def requests():
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as client:
            url = "/api/v1/investigations"
            headers = {"Idempotency-Key": "incident-1"}
            created = await client.post(url, json=PAYLOAD, headers=headers)
            repeated = await client.post(url, json=PAYLOAD, headers=headers)
            assert created.status_code == repeated.status_code == 202
            assert created.json() == repeated.json()
            inv_id = created.json()["id"]
            run_id = created.json()["run"]["id"]
            message_url = f"{url}/{inv_id}/messages"
            busy = await client.post(message_url, json={"message": "Explain"})
            assert busy.status_code == 409
            await client.post(f"/api/v1/runs/{run_id}/cancel")
            body = {"message": "Why did you conclude payment?",
                    "purpose": "explain"}
            headers = {"Idempotency-Key": "followup-1"}
            question = await client.post(
                message_url, json=body, headers=headers,
            )
            assert question.status_code == 202
            repeated_question = await client.post(
                message_url, json=body, headers=headers,
            )
            assert repeated_question.json() == question.json()
            conflicting = await client.post(message_url, json={
                **body, "purpose": "investigate",
            }, headers=headers)
            assert conflicting.status_code == 409
            history = await client.get(f"{url}/{inv_id}?limit=1")
            assert history.json()["next_offset"] == 1
            second = await client.get(f"{url}/{inv_id}?offset=1")
            turn = second.json()["turns"][0]
            assert turn["parent_run_id"] == run_id
            assert turn["user_message"] == body["message"]
            assert turn["purpose"] == "explain"
            assert question.json()["investigation_id"] == inv_id
            blank = await client.post(message_url, json={"message": "   "})
            partial = await client.post(message_url, json={
                "message": "new incident",
                "investigation_start": "2024-09-02T00:00:00Z",
            })
            wrong_purpose = await client.post(message_url, json={
                "message": "new incident", "purpose": "automatic",
            })
            assert blank.status_code == partial.status_code == 422
            assert wrong_purpose.status_code == 422
            missing = await client.get(f"{url}/{uuid4()}")
            assert missing.status_code == 404

    app.dependency_overrides[get_run_repository] = lambda: repository
    app.dependency_overrides[get_request_settings] = lambda: Settings(
        _env_file=None,
    )
    app.dependency_overrides[get_agent_runner] = lambda: object()
    try:
        asyncio.run(requests())
    finally:
        for dependency in (get_run_repository, get_request_settings,
                           get_agent_runner):
            app.dependency_overrides.pop(dependency)
    with sqlite3.connect(repository.path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM investigations"
        ).fetchone()[0]
        assert count == 1
        count = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert count == 2
