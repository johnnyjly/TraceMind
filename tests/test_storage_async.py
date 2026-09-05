"""Regression checks for nonblocking storage and durable admission order."""

import asyncio
from datetime import UTC, datetime, timedelta
import sqlite3
import threading

import httpx
from pydantic import ValidationError
import pytest

from tracemind.agent.runtime.harness import RunHarness
from tracemind.agent.storage.io import run_io
from tracemind.agent.storage.run_repository import RunRepository
from tracemind.api.app import (
    create_app,
    get_agent_runner,
    get_request_settings,
)
from tracemind.config import Settings
from tracemind.models import Incident
from tracemind.models.run import RunLimits, RunState, RunStatus


START = datetime(2024, 9, 1, tzinfo=UTC)


def state():
    return RunState(
        incident=Incident(title='failure', description='test incident'),
        limits=RunLimits(), allowed_tools=('query_metrics',),
        investigation_start=START,
        investigation_end=START + timedelta(hours=1),
    )


def test_sqlite_write_wait_does_not_block_event_loop(tmp_path):
    repository = RunRepository(tmp_path / 'runs.db')
    queued = state().model_copy(update={'status': RunStatus.QUEUED})
    locked, release = threading.Event(), threading.Event()

    def hold_lock():
        with sqlite3.connect(repository.path) as connection:
            connection.execute('BEGIN IMMEDIATE')
            locked.set()
            assert release.wait(3), 'event loop could not release SQLite lock'

    async def exercise():
        holder = asyncio.create_task(run_io(hold_lock))
        await asyncio.to_thread(locked.wait)
        write = asyncio.create_task(run_io(
            repository.create,
            queued,
            request_hash='locked'))
        try:
            await asyncio.sleep(.05)
            assert not write.done()
        finally:
            release.set()
        await holder
        return await write

    assert asyncio.run(exercise()).state.id == queued.id


def test_cancelled_storage_waits_for_write_before_terminal_cleanup():
    entered, release = threading.Event(), threading.Event()
    order = []

    def write():
        entered.set()
        assert release.wait(3)
        order.append('write')

    async def exercise():
        async def operation():
            try:
                await run_io(write)
            finally:
                order.append('cleanup')
        task = asyncio.create_task(operation())
        await asyncio.to_thread(entered.wait)
        task.cancel()
        await asyncio.sleep(.02)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(exercise())
    finally:
        release.set()
    assert order == ['write', 'cleanup']


def test_cancelled_storage_failure_preserves_cancellation_and_its_cause():
    entered, release = threading.Event(), threading.Event()

    def write():
        entered.set()
        assert release.wait(3)
        raise sqlite3.OperationalError('write failed')

    async def exercise():
        task = asyncio.create_task(run_io(write))
        await asyncio.to_thread(entered.wait)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert isinstance(caught.value.__cause__, sqlite3.OperationalError)

    try:
        asyncio.run(exercise())
    finally:
        release.set()


def test_parallel_tools_commit_reserved_budget_before_external_calls(tmp_path):
    repository = RunRepository(tmp_path / 'runs.db')
    queued = state().model_copy(update={'status': RunStatus.QUEUED})
    repository.create(queued, request_hash='parallel')
    claim = repository.claim_next()
    snapshots = []

    async def exercise():
        async def persist(current):
            await asyncio.sleep(.01)
            await run_io(
                repository.save_progress,
                current.id,
                claim.owner,
                current,
                [])
            snapshots.append(current)
        harness = RunHarness(claim.run.state, persist=persist)

        async def external():
            stored = await run_io(repository.get, queued.id)
            assert stored.state.tool_calls['query_metrics'] >= 1
            assert any(e.kind == 'tool.started' for e in stored.state.events)
            return []

        await asyncio.gather(*[
            harness.tool_call(
                'query_metrics',
                {'expression': str(i)},
                external)
            for i in range(2)
        ])

    asyncio.run(exercise())
    counts = [s.tool_calls.get('query_metrics', 0) for s in snapshots]
    assert counts == sorted(counts)
    assert repository.get(queued.id).state.tool_calls == {'query_metrics': 2}
    sequences = [len(s.events) for s in snapshots]
    assert sequences == sorted(sequences)


def test_failed_admission_commit_never_calls_provider():
    called = False

    async def exercise():
        async def persist(_state):
            raise sqlite3.OperationalError('synthetic write failure')

        async def model():
            nonlocal called
            called = True
        harness = RunHarness(state(), persist=persist)
        with pytest.raises(sqlite3.OperationalError):
            await harness.model_call(model, model_name='synthetic')

    asyncio.run(exercise())
    assert not called


def test_queue_has_no_model_dependency_and_initializes_store_once(
    tmp_path,
    monkeypatch,
):
    import tracemind.api.app as api
    application = create_app()
    settings = Settings(
        _env_file=None,
        agent_run_database=tmp_path / 'runs.db')
    application.dependency_overrides[get_request_settings] = lambda: settings

    def forbidden_runner():
        raise AssertionError('queue must not instantiate a model')
    application.dependency_overrides[get_agent_runner] = forbidden_runner
    opened = []
    real_repository = api.RunRepository

    def open_repository(path):
        opened.append(path)
        return real_repository(path)
    monkeypatch.setattr(api, 'RunRepository', open_repository)
    body = {
        'incident': {'title': 'failure', 'description': 'test'},
        'investigation_start': START.isoformat(),
        'investigation_end': (START + timedelta(hours=1)).isoformat(),
    }

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application),
            base_url='http://test',
        ) as client:
            replies = await asyncio.gather(*[
                client.post('/api/v1/runs', json=body) for _ in range(3)
            ])
            assert all(r.status_code == 202 for r in replies)

    asyncio.run(exercise())
    assert opened == [settings.agent_run_database]


def test_removed_transport_is_rejected_but_old_records_remain_readable():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, telemetry_transport='direct')
    assert RunState.model_validate(state().model_dump() | {
        'telemetry_transport': 'direct',
    }).telemetry_transport == 'direct'


def test_worker_shutdown_during_claim_finalizes_transferred_ownership(
    tmp_path,
    monkeypatch,
):
    from tracemind.agent.worker import RunWorker

    repository = RunRepository(tmp_path / 'runs.db')
    queued = state().model_copy(update={'status': RunStatus.QUEUED})
    repository.create(queued, request_hash='claim-cancel')
    entered, release = threading.Event(), threading.Event()
    original = repository.claim_next

    def delayed_claim():
        claim = original()
        entered.set()
        assert release.wait(3)
        return claim

    monkeypatch.setattr(repository, 'claim_next', delayed_claim)
    settings = Settings(_env_file=None, agent_run_database=repository.path,
                        agent_checkpoint_database=tmp_path / 'checkpoints.db')

    async def exercise():
        task = asyncio.create_task(RunWorker(
            settings,
            repository=repository).run_once())
        await asyncio.to_thread(entered.wait)
        task.cancel()
        await asyncio.sleep(.02)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(exercise())
    finally:
        release.set()
    stored = repository.get(queued.id)
    assert stored.state.status is RunStatus.CANCELLED
    assert stored.lease_until is None
    assert stored.state.events[-1].kind == 'run.finished'
