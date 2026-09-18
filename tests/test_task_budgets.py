"""Task allocations protect follow-up investigation under shared run limits."""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from test_harness import tool_call
from test_multi_agent import (
    DRAFT, REPORT, TASK, coordinator_packet, dispatch_tasks, response,
    run, service_for, submit,
)
from test_recovery import harness, http_failure

from tracemind.agent.runtime.harness import (
    RunHarness, TaskToolBudgetExhausted,
)
from tracemind.models.collaboration import (
    CollaborationState, InvestigationTask, TaskRecord,
)
from tracemind.models.run import RunLimits, RunState
from tracemind.telemetry.errors import TelemetryClientError


def allocate(run, **limits):
    run.state.collaboration = CollaborationState(tasks={
        key: TaskRecord(
            id=key, round=0, task=InvestigationTask(**TASK),
            status="running", tool_limit=limit,
        ) for key, limit in limits.items()
    })


async def in_task(run, key, operation):
    task_token = run._task_id.set(key)
    role_token = run._role.set("telemetry")
    try:
        return await operation()
    finally:
        run._role.reset(role_token)
        run._task_id.reset(task_token)


def test_parallel_calls_cannot_spend_sibling_or_followup_allocations():
    run = harness(tool_calls=20)
    allocate(run, owner=1, sibling=2)
    executed = 0

    async def backend():
        nonlocal executed
        executed += 1
        await asyncio.sleep(0)
        return []

    async def scenario():
        results = await asyncio.gather(*(
            in_task(run, "owner", lambda: run.tool_call(
                "query_metrics", {"expression": "same"}, backend,
            )) for _ in range(3)
        ), return_exceptions=True)
        assert sum(isinstance(r, TaskToolBudgetExhausted)
                   for r in results) == 2
        await in_task(run, "sibling", lambda: run.tool_call(
            "query_metrics", {"expression": "sibling"}, backend,
        ))

    asyncio.run(scenario())
    assert executed == run.state.backend_attempts == 2
    assert sum(run._signatures.values()) == 2
    assert run.state.collaboration.tasks["owner"].tool_calls == 1
    assert run.state.collaboration.tasks["sibling"].tool_calls == 1
    assert run.finalization_reason is None


def test_local_exhaustion_precedes_retry_backoff_and_global_deadline():
    run = harness(tool_calls=20, duration_seconds=10)
    allocate(run, owner=1, sibling=1)

    async def failing():
        raise TelemetryClientError("failed") from http_failure(
            503, retry_after="60",
        )

    async def succeeding():
        return []

    async def scenario():
        with pytest.raises(TaskToolBudgetExhausted) as error:
            await in_task(run, "owner", lambda: run.tool_call(
                "query_metrics", {"expression": "a"}, failing,
            ))
        assert error.value.task_id == "owner"
        await in_task(run, "sibling", lambda: run.tool_call(
            "query_metrics", {"expression": "b"}, succeeding,
        ))

    asyncio.run(scenario())
    assert run.finalization_reason is None
    assert not any(e.kind == "recovery.retry_scheduled"
                   for e in run.state.events)
    assert run.state.backend_attempts == 2


def test_shared_waiter_keeps_its_allocation_after_owner_retry_exhaustion():
    run = harness(tool_calls=20)
    allocate(run, owner=1, waiter=2)

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def failing():
            entered.set()
            await release.wait()
            raise TelemetryClientError("failed") from http_failure(503)

        async def success():
            return [{"result": "retained"}]

        def shared(backend):
            return run.shared_tool_call(
                "query_metrics", {"expression": "same"}, backend,
            )

        owner = asyncio.create_task(in_task(
            run, "owner", lambda: shared(failing),
        ))
        await entered.wait()
        waiter = asyncio.create_task(in_task(
            run, "waiter", lambda: shared(failing),
        ))
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(owner, waiter, return_exceptions=True)
        assert all(isinstance(r, TaskToolBudgetExhausted)
                   and r.task_id == "owner" for r in results)
        assert run.state.collaboration.tasks["waiter"].tool_calls == 0
        expected = await in_task(run, "waiter", lambda: shared(success))
        cached = await in_task(run, "owner", lambda: shared(success))
        assert cached == expected

    asyncio.run(scenario())
    assert run.state.backend_attempts == 2
    assert run.state.collaboration.tasks["waiter"].tool_calls == 1
    assert run.finalization_reason is None


def test_reservation_survives_interruption_before_backend_execution(tmp_path):
    run = harness(tool_calls=20)
    allocate(run, owner=1)
    saved = tmp_path / "reserved.json"
    executed = 0

    class InjectedCrash(BaseException):
        pass

    async def persist(state):
        if not saved.exists():
            saved.write_text(state.model_dump_json())
            raise InjectedCrash()

    async def backend():
        nonlocal executed
        executed += 1
        return []

    run._persist = persist

    async def scenario():
        with pytest.raises(InjectedCrash):
            await in_task(run, "owner", lambda: run.tool_call(
                "query_metrics", {"expression": "a"}, backend,
            ))
        restored = RunHarness(
            RunState.model_validate_json(saved.read_text()), resume=True,
        )
        with pytest.raises(TaskToolBudgetExhausted):
            await in_task(restored, "owner", lambda: restored.tool_call(
                "query_metrics", {"expression": "a"}, backend,
            ))
        assert restored.state.collaboration.tasks["owner"].tool_calls == 1
        assert restored.finalization_reason is None
        assert restored.state.tool_calls == {"query_metrics": 1}

    asyncio.run(scenario())
    assert executed == 0


def test_graph_keeps_followup_tools_after_initial_reports(monkeypatch):
    def respond(messages, calls):
        packet = coordinator_packet(messages)
        if packet is not None:
            if not packet["tasks"]:
                assert packet["batch_tool_budget"] == 10
                return dispatch_tasks({"tasks": [
                    {**TASK, "question": "first"},
                    {**TASK, "question": "second"},
                ]})
            if len(packet["task_history"]) == 2:
                assert packet["can_investigate"]
                assert packet["batch_tool_budget"] == 10
                return dispatch_tasks({"tasks": [
                    {**TASK, "question": "mechanism"},
                ]})
            return submit(DRAFT)
        if not any(isinstance(m, ToolMessage) for m in messages):
            task = json.loads(messages[1].text)
            count = 1 if task["goal"] == "mechanism" else 6
            calls = [tool_call(arguments={
                "expression": f"{task['goal']}-{i}",
            }).tool_calls[0] for i in range(count)]
            return AIMessage(content="", tool_calls=calls)
        return tool_call("InvestigationReport", REPORT)

    service, _, _, telemetry = service_for(
        monkeypatch, respond=respond,
        limits=RunLimits(model_calls=20, tool_calls=20),
    )
    state = run(service).state
    tasks = list(state.collaboration.tasks.values())
    assert [t.tool_limit for t in tasks] == [5, 5, 10]
    assert [t.tool_calls for t in tasks] == [5, 5, 1]
    assert all(t.status == "completed" and t.report.findings for t in tasks)
    assert len(telemetry.calls) == 11
    assert not any(e.kind == "policy.finalizing" for e in state.events)


@pytest.mark.parametrize("remaining, count", [(1, 1), (3, 1), (3, 2)])
def test_small_first_batch_exposes_and_enforces_admissible_task_count(
    monkeypatch, remaining, count,
):
    attempted = False

    def respond(messages, calls):
        nonlocal attempted
        packet = coordinator_packet(messages)
        if packet is not None and not packet["tasks"]:
            assert packet["max_parallel_tasks"] == 1
            number = count if not attempted else 1
            attempted = True
            return dispatch_tasks({"tasks": [
                {**TASK, "question": f"question-{i}"} for i in range(number)
            ]})
        return response(messages, calls)

    service, _, _, _ = service_for(
        monkeypatch, respond=respond,
        limits=RunLimits(model_calls=20, tool_calls=remaining),
    )
    state = run(service).state
    assert len(state.collaboration.tasks) == 1
    record = next(iter(state.collaboration.tasks.values()))
    assert record.tool_limit == record.tool_calls == 1
    if count == 2:
        assert any("task_allocation" in json.dumps(e.data)
                   for e in state.events)


def test_pre_allocation_checkpoint_cannot_reset_local_budget():
    with pytest.raises(ValidationError):
        CollaborationState.model_validate({"version": "graph-v5"})
