"""Persisted initial triage stays readable independently of model review."""

import asyncio
import json
import sqlite3
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
import pytest

from tracemind.agent.errors import DiagnosisCancelled, DiagnosisError
from tracemind.agent.storage.run_repository import RunRepository
from tracemind.agent.workflow.triage import current_triage
from tracemind.api.app import app, get_run_repository
from tracemind.api.progress import stream_progress
from tracemind.models.run import RunEvent, RunState, RunStatus
from test_verifier import FIXTURE, ReviewModel, diagnose, make_service, review


@pytest.mark.parametrize("outcome", ["supported", "insufficient", "truncated",
                                     "cancelled"])
def test_candidate_readable_while_review_blocked_then_updated(
    monkeypatch, tmp_path, outcome,
):
    entered, release = asyncio.Event(), asyncio.Event()

    class BlockedReview(ReviewModel):
        async def _agenerate(self, messages, **kwargs):
            entered.set()
            await release.wait()
            return await super()._agenerate(messages, **kwargs)

    model = BlockedReview(body=review(
        "insufficient" if outcome == "insufficient" else "supported"),
        mode="truncated" if outcome == "truncated" else "")
    service, _ = make_service(monkeypatch, model)
    repository = RunRepository(tmp_path / "runs.sqlite3")
    state = RunState(
        incident=FIXTURE.case.incident, investigation_start=FIXTURE.start,
        investigation_end=FIXTURE.end, limits=service._limits,
        allowed_tools=service._allowed_tools, status=RunStatus.QUEUED,
        verification_policy=service._verification_policy,
    )
    repository.create(state, request_hash=str(state.id))
    claim = repository.claim_next()
    url = f"/api/v1/runs/{state.id}"

    class Subscriber:
        async def is_disconnected(self):
            return False

    async def execute():
        task = asyncio.create_task(service.run_persisted(
            claim.run.state, [],
            persist=lambda state, evidence: repository.save_progress(
                state.id, claim.owner, state, evidence),
            is_cancelled=lambda: repository.should_stop(state.id, claim.owner),
            resume=False, graph_resume=False,
        ))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            async with AsyncClient(transport=ASGITransport(app=app),
                                   base_url="http://test") as client:
                initial = await client.get(url + "/triage")
                assert initial.status_code == 200, initial.text
                assert initial.headers["cache-control"] == "no-store"
                body = initial.json()
                assert body["status"] == "running"
                assert body["diagnosis"]["review_status"] == "pending"
                assert body["diagnosis"]["status"] == "inconclusive"
                assert all(i["evidence_support"] == "not_assessed"
                           for i in body["diagnosis"]["items"])
                assert (await client.get(url + "/result")).status_code == 409
                status = (await client.get(url)).json()
                assert status["triage_url"] == url + "/triage"
                reopened = RunRepository(repository.path).get(state.id)
                restored = current_triage(reopened.state, reopened.evidence)
                assert restored.review_status == "pending"
                # Reading/reconnecting does not own or cancel the worker task.
                frames = stream_progress(repository, state.id, Subscriber(),
                                         poll_seconds=.001)
                async for frame in frames:
                    if '"kind":"triage.available"' in frame:
                        event = json.loads(frame.split("data: ", 1)[1])
                        assert (event["data"]["candidate_id"]
                                == body["candidate_id"])
                        assert set(event["data"]) == {
                            "candidate_id", "triage_url", "review_status"}
                        break
                await frames.aclose()
                assert not task.done()
                assert not repository.get(state.id).cancel_requested
                if outcome == "cancelled":
                    repository.cancel(state.id)
                    pending_cancel = (await client.get(url + "/triage")).json()
                    assert (pending_cancel["diagnosis"]["review_status"]
                            == "unavailable")
                    task.cancel()
                else:
                    release.set()
                try:
                    result = await task
                    final_state, evidence = result.state, result.evidence
                except DiagnosisCancelled as exc:
                    final_state, evidence = exc.state, exc.evidence
                repository.finish(state.id, claim.owner, final_state, evidence)
                final = await client.get(url + "/triage")
                assert final.status_code == 200, final.text
                final_body = final.json()
                assert final_body["candidate_id"] == body["candidate_id"]
                diagnosis = final_body["diagnosis"]
                assert [i["text"] for i in diagnosis["items"]] == [
                    i["text"] for i in body["diagnosis"]["items"]]
                if outcome in {"truncated", "cancelled"}:
                    assert diagnosis["review_status"] == "unavailable"
                    assert all(i["review_status"] == "unreviewed"
                               for i in diagnosis["items"])
                else:
                    assert diagnosis["review_status"] == "completed"
                    assert diagnosis["items"][0]["evidence_support"] == outcome
                    terminal = (await client.get(url + "/result")).json()
                    assert diagnosis == terminal["diagnosis"]
                replay = await client.get(url + "/events", headers={
                    "Last-Event-ID": str(event["sequence"] - 1)})
                assert '"kind":"triage.available"' in replay.text
                assert "event: complete" in replay.text
                return reopened
        finally:
            release.set()
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    app.dependency_overrides[get_run_repository] = lambda: repository
    try:
        asyncio.run(execute())
    finally:
        app.dependency_overrides.pop(get_run_repository)


@pytest.mark.parametrize("changed", [
    "evidence", "candidate", "items", "rubric", "query", "archived",
    "review_text",
])
def test_changed_subject_cannot_reuse_old_review(monkeypatch, changed):
    service, _ = make_service(monkeypatch, ReviewModel(body=review()))
    result = diagnose(service)
    state, evidence = result.state, result.evidence
    assert current_triage(state, evidence).review_status == "completed"
    if changed == "evidence":
        evidence[0].summary += " changed"
    elif changed == "candidate":
        state.verification.candidate.root_cause += " changed"
    elif changed == "items":
        state.candidate_items[0].text += " changed"
    elif changed == "rubric":
        state.verification.rubric_version = "older-rubric"
    elif changed == "query":
        state.events.append(RunEvent(
            run_id=state.id, span_id=uuid4(), sequence=len(state.events) + 1,
            kind="tool.completed", data={"tool": "search_logs"},
        ))
    elif changed == "archived":
        # Reinvestigation clears the current record; history is not current.
        state.verification = None
    else:
        state.verification.review.claims[0].text += " changed"
    shown = current_triage(state, evidence)
    if changed == "review_text":
        assert shown.review_status == "unavailable"
        assert all(i.evidence_support == "not_assessed" for i in shown.items)
    else:
        assert shown is None


def test_expired_lease_cancel_retains_candidate(monkeypatch, tmp_path):
    service, _ = make_service(monkeypatch, ReviewModel(body=review()))
    result = diagnose(service)
    state = result.state
    state.status = RunStatus.QUEUED
    state.diagnosis = None
    state.verification.status = "running"
    state.verification.review = None
    repository = RunRepository(tmp_path / "runs.sqlite3")
    repository.create(state, request_hash=str(state.id))
    claim = repository.claim_next()
    repository.save_progress(
        state.id, claim.owner, claim.run.state, result.evidence,
    )
    with sqlite3.connect(repository.path) as connection:
        connection.execute("UPDATE runs SET lease_until=0 WHERE id=?",
                           (str(state.id),))
    cancelled = repository.cancel(state.id)
    shown = current_triage(cancelled.state, cancelled.evidence,
                           cancel_requested=cancelled.cancel_requested)
    assert cancelled.state.status is RunStatus.CANCELLED
    assert shown.review_status == "unavailable"
    assert shown.review_error_code == "cancelled"
    assert all(i.evidence_support == "not_assessed" for i in shown.items)


def test_resume_review_preserves_candidate_without_new_call(monkeypatch):
    model = ReviewModel(body=review())
    service, _ = make_service(monkeypatch, model)
    result = diagnose(service)
    state = result.state
    state.status = RunStatus.RUNNING
    state.diagnosis = None
    state.verification.status = "running"
    state.verification.review = None
    state.stop_reason = None
    state.finished_at = None
    # Reconstruct the durable checkpoint at admission, before review returns.
    admitted = next(e.sequence for e in state.events
                    if e.kind == "model.started"
                    and e.data.get("purpose") == "verification")
    state.events = [e for e in state.events if e.sequence <= admitted]
    assert current_triage(state, result.evidence).review_status == "pending"
    calls = model.calls

    async def resume():
        try:
            await service.run_persisted(
                state, result.evidence, persist=lambda *_: None,
                is_cancelled=lambda: False, resume=True, graph_resume=False,
            )
        except DiagnosisError:
            pass

    asyncio.run(resume())
    assert model.calls == calls
    assert state.verification.error_code == "ambiguous_attempt"
    shown = current_triage(state, result.evidence)
    assert shown.review_status == "unavailable"
    assert shown.review_error_code == "ambiguous_attempt"
    assert all(i.evidence_support == "not_assessed" for i in shown.items)


def test_no_candidate_and_unknown_run_are_distinct(tmp_path):
    from test_persistence import queued_state

    repository = RunRepository(tmp_path / "runs.sqlite3")
    state = queued_state()
    repository.create(state, request_hash=str(state.id))

    async def requests():
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as client:
            empty = await client.get(f"/api/v1/runs/{state.id}/triage")
            missing = await client.get(f"/api/v1/runs/{uuid4()}/triage")
            assert empty.status_code == 409
            assert missing.status_code == 404

    app.dependency_overrides[get_run_repository] = lambda: repository
    try:
        asyncio.run(requests())
    finally:
        app.dependency_overrides.pop(get_run_repository)


def test_cancel_after_candidate_checkpoint_before_reviewer_call(monkeypatch):
    model = ReviewModel(body=review())
    service, _ = make_service(monkeypatch, model)
    state = RunState(
        incident=FIXTURE.case.incident, investigation_start=FIXTURE.start,
        investigation_end=FIXTURE.end, limits=service._limits,
        allowed_tools=service._allowed_tools,
        verification_policy=service._verification_policy,
    )
    saved = []
    cancel = False

    def persist(current, evidence):
        nonlocal cancel
        if any(e.kind == "triage.available" for e in current.events):
            saved.append((current.model_copy(deep=True), list(evidence)))
            cancel = True

    async def execute():
        try:
            result = await service.run_persisted(
                state, [], persist=persist, is_cancelled=lambda: cancel,
                resume=False, graph_resume=False,
            )
            return result.state, result.evidence
        except (DiagnosisCancelled, DiagnosisError) as exc:
            return exc.state, exc.evidence

    final_state, evidence = asyncio.run(execute())
    assert saved and model.calls == 0
    assert final_state.status is RunStatus.CANCELLED
    shown = current_triage(final_state, evidence, cancel_requested=True)
    assert shown.review_status == "unavailable"
    assert all(i.evidence_support == "not_assessed" for i in shown.items)
