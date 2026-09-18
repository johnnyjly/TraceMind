"""Parallel admission, ambiguous charges and durable token reservations."""

import asyncio

import httpx
import pytest
from langchain_core.messages import AIMessage

from test_recovery import harness
from test_task_budgets import allocate, in_task
from tracemind.agent.runtime.harness import RunHarness, RunStopped
from tracemind.agent.runtime.recovery import ModelCallFailed
from tracemind.models.run import RunState, StopReason


def message(total=3):
    return AIMessage(content="ok", usage_metadata={
        "input_tokens": total - 1, "output_tokens": 1, "total_tokens": total,
    })


def test_parallel_admission_reserves_global_and_task_before_provider():
    run = harness(total_tokens=15)
    allocate(run, a=3, b=3)
    for task in run.state.collaboration.tasks.values():
        task.token_limit = 15

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def provider():
            entered.set()
            await release.wait()
            return message()

        first = asyncio.create_task(in_task(run, "a", lambda: run.model_call(
            provider, model_name="fake", estimated_tokens=10,
        )))
        await entered.wait()
        assert run.state.tokens.reserved == 10
        with pytest.raises(ModelCallFailed, match="token_budget"):
            await in_task(run, "b", lambda: run.model_call(
                provider, model_name="fake", estimated_tokens=10,
            ))
        release.set()
        await first

    asyncio.run(scenario())
    assert run.state.model_calls == 1
    assert run.state.tokens.model_dump() == {
        "used": 3, "reserved": 0, "unknown": 0,
    }
    assert run.state.collaboration.tasks["a"].tokens.used == 3
    assert run.state.collaboration.tasks["b"].tokens.committed == 0
    assert run.finalization_reason is None


def test_same_task_parallel_auxiliary_calls_share_allocation():
    run = harness(total_tokens=100)
    allocate(run, a=3)
    run.state.collaboration.tasks["a"].token_limit = 10

    async def provider():
        await asyncio.sleep(0.01)
        return message()

    async def scenario():
        return await asyncio.gather(*[
            in_task(run, "a", lambda: run.model_call(
                provider, model_name="fake", purpose="knowledge.retrieval",
                estimated_tokens=10,
            )) for _ in range(2)
        ], return_exceptions=True)

    results = asyncio.run(scenario())
    assert sum(isinstance(r, ModelCallFailed) for r in results) == 1
    assert run.state.model_calls == 1


def test_retry_keeps_unknown_charge_and_reestimates_actual_attempt():
    run = harness(total_tokens=40)
    estimates = iter([10, 20])
    calls = 0

    async def provider():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("unknown remote outcome")
        return message()

    asyncio.run(run.model_call(
        provider, model_name="fake", estimated_tokens=lambda: next(estimates),
    ))
    assert run.state.model_calls == calls == 2
    assert run.state.tokens.model_dump() == {
        "used": 3, "reserved": 0, "unknown": 10,
    }


def test_missing_usage_and_actual_overrun_are_never_refunded_as_zero():
    run = harness(total_tokens=20)

    async def unknown():
        return AIMessage(content="unknown")

    async def overrun():
        return message(15)

    async def scenario():
        await run.model_call(unknown, model_name="fake", estimated_tokens=10)
        await run.model_call(overrun, model_name="fake", estimated_tokens=5)
        with pytest.raises(RunStopped) as stopped:
            await run.model_call(overrun, model_name="fake",
                                 estimated_tokens=1)
        assert stopped.value.reason is StopReason.TOKEN_BUDGET

    asyncio.run(scenario())
    assert run.state.tokens.used == 15
    assert run.state.tokens.unknown == 10
    assert run.state.tokens.reserved == 0
    assert any(e.kind == "budget.estimate_exceeded" for e in run.state.events)


def test_crash_reservation_uses_existing_checkpoints_and_restores_once():
    run = harness(total_tokens=100)
    allocate(run, a=2)
    run.state.collaboration.tasks["a"].token_limit = 40
    snapshots = []

    async def persist(state):
        snapshots.append(state.model_dump_json())

    async def provider():
        assert RunState.model_validate_json(snapshots[0]).tokens.reserved == 10
        return message()

    run._persist = persist
    asyncio.run(in_task(run, "a", lambda: run.model_call(
        provider, model_name="fake", estimated_tokens=10,
    )))
    assert len(snapshots) == 2
    restored = RunHarness(RunState.model_validate_json(snapshots[0]),
                          resume=True)
    assert restored.state.tokens.unknown == 10
    assert restored.state.collaboration.tasks["a"].tokens.unknown == 10
    again = RunHarness(restored.state.model_copy(deep=True), resume=True)
    assert again.state.tokens.unknown == 10
    assert again.state.tokens.reserved == 0


def test_cap_cannot_be_used_without_admission_estimate():
    run = harness(total_tokens=1)

    async def provider():
        pytest.fail("provider must not be called")

    with pytest.raises(ValueError, match="token estimate"):
        asyncio.run(run.model_call(provider, model_name="fake"))
    assert run.state.model_calls == 0


def test_optional_retrieval_rejection_leaves_synthesis_reserve_usable():
    run = harness(total_tokens=100)

    async def provider():
        return message()

    async def scenario():
        with pytest.raises(ModelCallFailed, match="token_budget"):
            await run.model_call(
                provider, model_name="fake", purpose="knowledge.retrieval",
                required=False, estimated_tokens=60, protected_tokens=50,
            )
        assert run.state.model_calls == 0
        await run.model_call(provider, model_name="fake", estimated_tokens=90)

    asyncio.run(scenario())
    assert run.state.tokens.used == 3


def test_dispatch_reserve_covers_verifier_repair(monkeypatch):
    from test_multi_agent import run as execute, service_for
    from test_verifier import ReviewModel, review
    from tracemind.models.run import RunLimits

    limits = RunLimits(model_calls=12, total_tokens=1000000)
    reviewer = ReviewModel(body=review(), mode="repair")
    service, _, _, _ = service_for(
        monkeypatch, limits=limits, reviewer=reviewer,
    )
    state = execute(service).state
    record = next(iter(state.collaboration.tasks.values()))
    policy = state.verification_policy
    protected = (2 * (policy.input_bytes + 1024 + policy.output_tokens)
                 + (1 + limits.diagnosis_repairs) * limits.context_tokens)
    # The scripted first coordinator used 15 tokens before dispatch.
    assert record.token_limit == limits.total_tokens - 15 - protected
    assert record.tokens.used == 30
    assert state.tokens.used == state.usage.reported_total_tokens == 120
    assert state.tokens.reserved == state.tokens.unknown == 0
    assert reviewer.calls == 2


def test_coordinator_closes_readers_before_using_synthesis_reserve():
    from tracemind.agent.runtime.collaboration import CoordinatorDecision
    from langchain.agents.middleware import ModelRequest, ModelResponse
    from langchain_core.language_models.fake_chat_models import (
        FakeListChatModel,
    )
    from langchain_core.messages import HumanMessage
    from tracemind.agent.runtime.middleware import HarnessMiddleware

    run = harness(total_tokens=32769, diagnosis_repairs=0)
    middleware = HarnessMiddleware(
        run, output_name="CoordinatorDecision", investigation=False,
    )

    async def provider(request):
        assert request.tools == []
        assert "Choose submit with a proposal" in request.messages[-1].text
        return ModelResponse(result=[message()], structured_response=(
            CoordinatorDecision.model_validate({
                "action": "submit", "reason": "Finalize within budget",
                "proposal": {
                    "status": "inconclusive", "confidence": 0,
                    "findings": [], "hypotheses": [], "proposed_checks": [],
                    "limitations": [{"text": "Cause remains unestablished"}],
                },
            })
        ))

    asyncio.run(middleware.awrap_model_call(ModelRequest(
        model=FakeListChatModel(responses=["unused"]),
        messages=[HumanMessage(content="Produce final diagnosis")],
        tools=[{"type": "function", "function": {
            "name": "read_evidence", "description": "Read retained records",
            "parameters": {"type": "object", "properties": {}},
        }}],
    ), provider))
    assert run.finalization_reason is StopReason.TOKEN_BUDGET
    assert run.state.tokens.used == 3
