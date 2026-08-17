"""Verify durable SSE replay, privacy, disconnect and terminal boundaries."""

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from httpx import ASGITransport, AsyncClient

from tracemind.agent.persistence import RunRepository
from tracemind.agent.service import DiagnosisDraft, DiagnosisService
from tracemind.agent.worker import RunWorker
from tracemind.api.app import app, get_run_repository
from tracemind.api.progress import readable_event, stream_progress
from tracemind.config import Settings
from tracemind.models import Evidence, EvidenceSource, Incident
from tracemind.models.run import (
    RunEvent, RunLimits, RunState, RunStatus,
)


START = datetime(2024, 9, 1, tzinfo=UTC)


def queued(repository):
    state = RunState(
        incident=Incident(title="failure", description="investigate"),
        investigation_start=START,
        investigation_end=START + timedelta(hours=1),
        limits=RunLimits(), allowed_tools=("search_traces",),
        status=RunStatus.QUEUED,
    )
    repository.create(state, request_hash=str(state.id))
    return state


def test_sse_replay_cursor_validation_and_early_cancel(tmp_path):
    repository = RunRepository(tmp_path / "runs.sqlite3")
    state = queued(repository)
    repository.cancel(state.id)

    async def requests():
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as client:
            url = f"/api/v1/runs/{state.id}/events"
            all_events = await client.get(url)
            resumed = await client.get(url, headers={"Last-Event-ID": "1"})
            terminal = await client.get(url + "?after=2")
            bad = await client.get(url, headers={"Last-Event-ID": "-1"})
            future = await client.get(url + "?after=1000")
            missing = await client.get(f"/api/v1/runs/{uuid4()}/events")
            return all_events, resumed, terminal, bad, future, missing

    app.dependency_overrides[get_run_repository] = lambda: repository
    try:
        events, resumed, terminal, bad, future, missing = (
            asyncio.run(requests())
        )
    finally:
        app.dependency_overrides.pop(get_run_repository)
    assert events.status_code == 200
    assert events.headers["content-type"].startswith("text/event-stream")
    assert '"kind":"run.queued"' in events.text
    assert '"kind":"run.finished"' in events.text
    assert "event: complete" in events.text
    assert '"result_url": null' in events.text
    assert '"kind":"run.queued"' not in resumed.text
    assert '"kind":"run.finished"' in resumed.text
    assert "event: progress" not in terminal.text
    assert (bad.status_code, future.status_code, missing.status_code) == (
        400, 409, 404,
    )


def test_progress_omits_raw_payloads_and_internal_events():
    run_id = uuid4()
    event = RunEvent(run_id=run_id, span_id=uuid4(), sequence=1,
                     kind="tool.started", data={
                         "tool": "search_logs", "service": "payment",
                         "expression": "secret-query", "logs": "private",
                         "arguments_sha256": "hash-only",
                     })
    progress = readable_event(event)
    assert "payment" in progress.message
    assert "服务日志" in progress.message
    assert progress.data == {"tool": "search_logs", "service": "payment"}
    assert "private" not in progress.model_dump_json()
    event.kind = "context.admitted"
    assert readable_event(event) is None


def test_progress_arrives_during_execution_and_disconnect_keeps_run(tmp_path):
    repository = RunRepository(tmp_path / "runs.sqlite3")
    state = queued(repository)
    entered, release = asyncio.Event(), asyncio.Event()

    class SlowTelemetry:
        async def search_traces(self, service, *, start, end):
            entered.set()
            await release.wait()
            return [Evidence(source=EvidenceSource.TRACE,
                             observed_at=end, summary="a trace")]

    class Runner:
        async def run(self, _prompt, tools, **_kwargs):
            await tools[0].ainvoke({"service": "checkout"})
            return DiagnosisDraft(
                status="inconclusive", root_cause="No confirmed cause",
                confidence=0,
            )

    class Subscriber:
        disconnected = False

        async def is_disconnected(self):
            return self.disconnected

    telemetry = SlowTelemetry()
    settings = Settings(_env_file=None,
                        agent_run_database=repository.path,
                        agent_checkpoint_database=tmp_path / "graph.sqlite3")
    worker = RunWorker(
        settings, repository=repository,
        service_factory=lambda _saver, _client: DiagnosisService(
            telemetry, telemetry, telemetry, Runner(), event_directory=None,
        ),
    )

    async def execute():
        task = asyncio.create_task(worker.run_once())
        await entered.wait()
        subscriber = Subscriber()
        stream = stream_progress(repository, state.id, subscriber,
                                 poll_seconds=.001)
        delivered = []
        async for frame in stream:
            delivered.append(frame)
            if '"kind":"tool.started"' in frame:
                break
        assert not task.done()
        subscriber.disconnected = True
        async for _frame in stream:
            raise AssertionError("disconnected subscriber should stop")
        assert not repository.get(state.id).cancel_requested
        release.set()
        assert await task
        return "".join(delivered)

    frames = asyncio.run(execute())
    assert '"service":"checkout"' in frames
    assert repository.get(state.id).state.status is RunStatus.INCONCLUSIVE
    assert len(repository.get(state.id).evidence) == 1


def test_expired_cancelled_worker_gets_terminal_event(tmp_path):
    repository = RunRepository(tmp_path / "runs.sqlite3")
    state = queued(repository)
    repository.claim_next()
    repository.cancel(state.id)
    with sqlite3.connect(repository.path) as connection:
        connection.execute("UPDATE runs SET lease_until=0 WHERE id=?",
                           (str(state.id),))
    assert repository.claim_next() is None
    snapshot = repository.event_snapshot(state.id)
    assert snapshot.status is RunStatus.CANCELLED
    assert snapshot.events[-1].kind == "run.finished"
    assert json.loads(snapshot.events[-1].model_dump_json())["sequence"] == 2


def test_quiet_stream_heartbeats_and_finishes_after_cancel(tmp_path):
    repository = RunRepository(tmp_path / "runs.sqlite3")
    state = queued(repository)

    class Subscriber:
        async def is_disconnected(self):
            return False

    async def execute():
        stream = stream_progress(repository, state.id, Subscriber(),
                                 poll_seconds=.001, heartbeat_seconds=0)
        assert "run.queued" in await anext(stream)
        assert await anext(stream) == ": keep-alive\n\n"
        repository.cancel(state.id)
        assert "run.finished" in await anext(stream)
        assert "event: complete" in await anext(stream)
        try:
            await anext(stream)
        except StopAsyncIteration:
            return
        raise AssertionError("terminal streams must close")

    asyncio.run(execute())


def test_worker_setup_failure_is_visible_and_closes_stream(tmp_path):
    from tracemind.agent.errors import ModelConfigurationError
    repository = RunRepository(tmp_path / "runs.sqlite3")
    state = queued(repository)

    def unavailable(_saver, _client):
        raise ModelConfigurationError("private provider configuration")

    settings = Settings(_env_file=None,
                        agent_run_database=repository.path,
                        agent_checkpoint_database=tmp_path / "graph.sqlite3")
    worker = RunWorker(settings, repository=repository,
                       service_factory=unavailable)
    assert asyncio.run(worker.run_once())
    snapshot = repository.event_snapshot(state.id)
    assert snapshot.status is RunStatus.FAILED
    event = readable_event(snapshot.events[-1])
    assert event.message == "调查失败"
    assert event.data["reason"] == "model_error"
    assert "private provider" not in event.model_dump_json()


def test_cancel_racing_final_commit_preserves_completed_result(tmp_path):
    from tracemind.agent.harness import RunHarness
    from tracemind.models import Diagnosis
    from tracemind.models.run import StopReason
    repository = RunRepository(tmp_path / "runs.sqlite3")
    state = queued(repository)
    claim = repository.claim_next()
    item = Evidence(source=EvidenceSource.TRACE, observed_at=START,
                    summary="payment failed")
    harness = RunHarness(
        claim.run.state,
        persist=lambda current: repository.save_progress(
            state.id, claim.owner, current, [item],
        ),
    )
    claim.run.state.diagnosis = Diagnosis(
        incident_id=state.incident.id, status="complete",
        root_cause="Payment failed", confidence=.8, evidence_ids=[item.id],
    )
    harness.finish(RunStatus.SUCCEEDED, StopReason.COMPLETED)
    finalized = repository.cancel(state.id)
    assert finalized.state.status is RunStatus.SUCCEEDED
    assert finalized.result.diagnosis.evidence_ids == [item.id]
    assert finalized.state.events[-1].data["status"] == "succeeded"
    assert repository.event_snapshot(state.id).has_result
