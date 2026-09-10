"""Exercise cache accounting and stable prefixes through provider requests."""

import asyncio
import json

import httpx
from langchain_core.messages import AIMessage
from langchain_openai.chat_models.base import BaseChatOpenAI
import pytest

from proposal_fixtures import proposal_payload
from test_harness import (
    END, INCIDENT, START, ScriptedModel, Telemetry, diagnosis, make_service,
    run_service, tool_call,
)
from test_recovery import assert_tool_pairs, harness, http_failure
from tool_fixtures import FixtureToolClient

from tracemind.agent.runtime.runner import LangChainAgentRunner
from tracemind.agent.service import DiagnosisService
from tracemind.evaluation.evaluators import diagnosis_scores
from tracemind.evaluation.models import ExperimentOutput
from tracemind.models.run import RunLimits, RunState, StopReason


def provider_reply(message, usage):
    return httpx.Response(200, json={
        "id": "completion-offline", "object": "chat.completion",
        "created": 0, "model": "deepseek-flash",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": message}],
        "usage": usage,
    })


def provider_model(client):
    return BaseChatOpenAI(
        model="deepseek-flash", api_key="offline-test",
        base_url="https://api.deepseek.com", http_async_client=client,
        use_responses_api=False, max_retries=0,
        profile={"tool_calling": True, "structured_output": False},
    )


@pytest.mark.parametrize("fields, expected_hit", [
    ({"prompt_tokens_details": {"cached_tokens": 768}}, 768),
    ({"prompt_cache_hit_tokens": 768, "prompt_cache_miss_tokens": 232}, 768),
    ({"prompt_tokens_details": {"cached_tokens": 0}}, 0),
    ({}, None),
    ({"prompt_cache_hit_tokens": 1001}, None),
])
def test_provider_cache_usage_is_measured_or_explicitly_unknown(
    fields, expected_hit,
):
    run = harness()
    usage = {"prompt_tokens": 1000, "completion_tokens": 20,
             "total_tokens": 1020, "private_debug": "never-retain", **fields}

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _: provider_reply(
                {"role": "assistant", "content": "done"}, usage),
        )) as client:
            model = provider_model(client)
            await run.model_call(lambda: model.ainvoke("offline request"),
                                 model_name="deepseek-flash",
                                 purpose="verification")

    asyncio.run(scenario())
    event = next(e for e in run.state.events if e.kind == "model.completed")
    known = expected_hit is not None
    assert event.data["cache_usage_reported"] is known
    assert event.data["cache_hit_tokens"] == expected_hit
    assert event.data["cache_miss_tokens"] == (
        1000 - expected_hit if known else None)
    assert event.data["model"] == "deepseek-flash"
    assert event.data["purpose"] == "verification"
    assert event.data["call"] == 1
    assert run.state.usage.reported_total_tokens == 1020
    assert run.state.usage.reported_cache_model_calls == int(known)
    assert run.state.usage.unreported_cache_model_calls == int(not known)
    assert "never-retain" not in run.state.model_dump_json()


def test_cache_totals_survive_storage_and_rates_use_reported_tokens():
    run = harness()

    async def scenario():
        for total, hit in [(1000, 768), (2000, 0), (500, None)]:
            response = AIMessage(content="done", usage_metadata={
                "input_tokens": total, "output_tokens": 20,
                "total_tokens": total + 20,
                "input_token_details": {} if hit is None else {
                    "cache_read": hit},
            })

            async def call():
                return response

            await run.model_call(call, model_name="test")

    asyncio.run(scenario())
    state = RunState.model_validate_json(run.state.model_dump_json())
    assert state.usage.reported_cache_hit_tokens == 768
    assert state.usage.reported_cache_miss_tokens == 2232
    assert state.usage.reported_cache_model_calls == 2
    assert state.usage.unreported_cache_model_calls == 1
    scores = {s.name: s.value for s in diagnosis_scores(
        output=ExperimentOutput(run_state=state))}
    assert scores["cache_hit_rate"] == 768 / 3000
    assert scores["cache_usage_coverage"] == 2 / 3
    assert scores["cache_hit_tokens"] == 768
    assert scores["cache_miss_tokens"] == 2232
    legacy = state.model_dump()
    legacy["usage"] = {"reported_total_tokens": 3560}
    legacy_scores = {s.name: s.value for s in diagnosis_scores(
        output=ExperimentOutput(run_state=RunState.model_validate(legacy)))}
    assert "cache_hit_rate" not in legacy_scores
    assert "cache_hit_tokens" not in legacy_scores
    assert legacy_scores["cache_usage_coverage"] == 0


@pytest.mark.parametrize("finalizing", [False, True])
def test_wire_messages_keep_previous_request_prefix(tmp_path, finalizing):
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            name, args = "query_metrics", {"expression": "up"}
        else:
            observation = next(m for m in payload["messages"]
                               if m["role"] == "tool")
            name, args = "DiagnosisProposal", {
                "status": "complete", "root_cause": "Payment Charge failed",
                "confidence": .8,
                "evidence_ids": [json.loads(observation["content"])[0]["id"]],
            }
        return provider_reply({
            "role": "assistant", "content": "", "tool_calls": [{
                "id": f"call-{len(requests)}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(
                    proposal_payload(args) if name == "DiagnosisProposal"
                    else args)},
            }],
        }, {"prompt_tokens": 1000, "completion_tokens": 20,
            "total_tokens": 1020,
            "prompt_tokens_details": {"cached_tokens": 768}})

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond),
        ) as client:
            return await make_service(
                provider_model(client), Telemetry(), tmp_path,
                limits=RunLimits(tool_calls=1 if finalizing else 12),
            ).diagnose_with_evidence(INCIDENT, start=START, end=END)

    result = asyncio.run(scenario())
    first, second = requests
    assert second["messages"][:len(first["messages"])] == first["messages"]
    assert first["messages"][0] == second["messages"][0]
    assert "Remaining budget" not in first["messages"][0]["content"]
    snapshots = [m for m in second["messages"]
                 if m.get("name") == "runtime_budget"]
    assert len(snapshots) == 2
    assert "8 model calls" in snapshots[0]["content"]
    assert "7 model calls" in snapshots[1]["content"]
    if finalizing:
        assert [t["function"]["name"] for t in second["tools"]] == [
            "DiagnosisProposal"]
        assert "Investigation is closed" in snapshots[-1]["content"]
    else:
        assert first["tools"] == second["tools"]
    assert result.state.stop_reason is StopReason.COMPLETED
    assert result.state.usage.reported_cache_hit_tokens == 1536
    assert result.state.usage.reported_cache_model_calls == 2


@pytest.mark.parametrize("failure", ["retry", "repair", "fallback"])
def test_snapshots_preserve_history_through_recovery(tmp_path, failure):
    def respond(messages, calls):
        assert_tool_pairs(messages)
        if calls == 1:
            if failure == "repair":
                return tool_call("DiagnosisProposal", {"status": "invalid"})
            raise http_failure(503)
        if calls == 2:
            return tool_call()
        return diagnosis(messages)

    model = ScriptedModel(respond=respond)
    telemetry = Telemetry()
    service = DiagnosisService(
        LangChainAgentRunner(
            model, fallback_model=model if failure == "fallback" else None),
        tool_client=FixtureToolClient(telemetry, telemetry, telemetry),
        event_directory=tmp_path,
        limits=RunLimits(retry_attempts=0 if failure == "fallback" else 1,
                         retry_backoff_seconds=0),
    )
    result = run_service(service)
    assert result.state.stop_reason is StopReason.COMPLETED
    assert model.calls == 3
    assert result.state.diagnosis_repairs == int(failure == "repair")
    assert result.state.model_switches == int(failure == "fallback")
    for previous, current in zip(model.inputs, model.inputs[1:]):
        assert current[:len(previous)] == previous
    snapshots = [m for m in model.inputs[-1] if m.name == "runtime_budget"]
    assert len(snapshots) == len({m.id for m in snapshots}) == 3
    assert [f"{remaining} model calls" in message.text
            for remaining, message in zip((8, 7, 6), snapshots)] == [True] * 3
    assert result.state.usage.unreported_cache_model_calls == 3


def test_concurrent_cache_calls_keep_their_admission_number():
    run = harness()

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        response = AIMessage(content="done", usage_metadata={
            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
            "input_token_details": {"cache_read": 1},
        })

        async def first():
            entered.set()
            await release.wait()
            return response

        async def second():
            await entered.wait()
            release.set()
            return response

        await asyncio.gather(
            run.model_call(first, model_name="test", purpose="knowledge"),
            run.model_call(second, model_name="test", purpose="verification"),
        )

    asyncio.run(scenario())
    starts = {e.span_id: e.data["call"] for e in run.state.events
              if e.kind == "model.started"}
    for event in run.state.events:
        if event.kind == "model.completed":
            assert event.data["call"] == starts[event.span_id]
    assert run.state.usage.reported_cache_hit_tokens == 2
    assert run.state.usage.reported_cache_model_calls == 2
