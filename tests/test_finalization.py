"""Exercise tool closure and evidence-grounded output in the real loop."""

from proposal_fixtures import proposal_payload

import asyncio
import json
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai.chat_models.base import BaseChatOpenAI

from test_harness import (
    END, INCIDENT, START, ScriptedModel, Telemetry, make_service,
    run_service, tool_call,
)
from test_recovery import assert_tool_pairs

from tracemind.agent.runtime.harness import (
    InvestigationClosed,
    RunHarness,
    RunStopped,
)
from tracemind.api.progress import readable_event
from tracemind.models.run import RunLimits, RunState, RunStatus, StopReason


def citation(messages):
    """Find a real observation while ignoring blocked-tool error records."""
    for message in messages:
        if isinstance(message, ToolMessage):
            content = json.loads(message.content)
            if isinstance(content, list) and content and "id" in content[0]:
                return content[0]["id"]
    raise AssertionError("the test expected an observed citation")


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("reason, limits", [
    (StopReason.TOOL_BUDGET, RunLimits(tool_calls=1)),
    (StopReason.REPEATED_TOOL_CALL, RunLimits(identical_tool_calls=1)),
    (StopReason.NO_PROGRESS, RunLimits(no_progress_rounds=1)),
])
def test_soft_limits_close_tools_and_accept_valid_final_diagnosis(
    tmp_path, native, complete, reason, limits,
):
    def respond(messages, calls):
        assert_tool_pairs(messages)
        if "Investigation is closed" not in messages[-1].text:
            return tool_call()
        assert set(model.bindings[-1]) <= {"DiagnosisProposal"}
        fields = {
            "status": "complete" if complete else "inconclusive",
            "root_cause": ("Payment Charge failed" if complete else
                           "Payment errors observed; cause needs checking"),
            "confidence": .8 if complete else .3,
            "evidence_ids": [citation(messages)],
            "remediation": ["Check the payment service"],
        }
        return (AIMessage(content=json.dumps(fields), usage_metadata={
            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
        }) if native else tool_call("DiagnosisProposal", fields))

    model = ScriptedModel(respond=respond, profile={
        "structured_output": native, "tool_calling": True,
    })
    telemetry = Telemetry()
    run = run_service(make_service(model, telemetry, tmp_path, limits=limits))
    assert run.state.status is (RunStatus.SUCCEEDED if complete else
                                RunStatus.INCONCLUSIVE)
    assert run.state.stop_reason is (StopReason.COMPLETED if complete else
                                     StopReason.INSUFFICIENT_EVIDENCE)
    assert run.diagnosis.evidence_ids == [run.evidence[0].id]
    assert "Investigation stopped" not in run.diagnosis.root_cause
    phases = [e for e in run.state.events if e.kind == "policy.finalizing"]
    assert len(phases) == 1
    assert phases[0].data == {"reason": reason.value}
    assert readable_event(phases[0]).phase == "finalizing"
    assert model.calls == (2 if reason is StopReason.TOOL_BUDGET else 3)
    expected_tools = 2 if reason is StopReason.NO_PROGRESS else 1
    assert len(telemetry.calls) == expected_tools
    assert run.state.usage.reported_model_calls == model.calls
    assert run.state.diagnosis_repairs == 0


def test_finalization_repairs_fake_citation_without_tools(tmp_path):
    def respond(messages, calls):
        assert_tool_pairs(messages)
        if calls == 1:
            return tool_call()
        assert model.bindings[-1] == ["DiagnosisProposal"]
        return tool_call("DiagnosisProposal", {
            "status": "complete", "root_cause": "Payment Charge failed",
            "confidence": .8,
            "evidence_ids": [
                str(uuid4()) if calls == 2 else citation(messages),
            ],
        })

    model, telemetry = ScriptedModel(respond=respond), Telemetry()
    run = run_service(make_service(
        model, telemetry, tmp_path, limits=RunLimits(tool_calls=1),
    ))
    assert run.state.status is RunStatus.SUCCEEDED
    assert run.diagnosis.evidence_ids == [run.evidence[0].id]
    assert run.state.diagnosis_repairs == 1
    assert len(telemetry.calls) == 1
    assert model.calls == 3


def test_finalization_cannot_exceed_remaining_model_attempts(tmp_path):
    model = ScriptedModel(respond=lambda messages, calls: tool_call())
    telemetry = Telemetry()
    run = run_service(make_service(
        model, telemetry, tmp_path,
        limits=RunLimits(model_calls=1, tool_calls=1),
    ))
    assert model.calls == 1
    assert telemetry.calls == []
    assert model.bindings == [["DiagnosisProposal"]]
    assert run.state.stop_reason is StopReason.MODEL_BUDGET
    assert [e.data["reason"] for e in run.state.events
            if e.kind == "policy.finalizing"] == ["model_budget"]


@pytest.mark.parametrize("hard_stop", [
    StopReason.MODEL_BUDGET, StopReason.DEADLINE, StopReason.CANCELLED,
])
def test_finalizing_run_still_obeys_hard_admission_checks(hard_stop):
    state = RunState(
        incident=INCIDENT, investigation_start=START, investigation_end=END,
        limits=RunLimits(), allowed_tools=("query_metrics",),
    )
    cancelled = False
    run = RunHarness(state, is_cancelled=lambda: cancelled)
    run.begin_finalization(StopReason.NO_PROGRESS)
    if hard_stop is StopReason.MODEL_BUDGET:
        state.model_calls = state.limits.model_calls
    elif hard_stop is StopReason.DEADLINE:
        state.deadline_at = START
    else:
        cancelled = True
    called = False

    async def model():
        nonlocal called
        called = True

    with pytest.raises(RunStopped) as caught:
        asyncio.run(run.model_call(model, model_name="test"))
    assert caught.value.reason is hard_stop
    assert not called


def test_finalization_phase_survives_state_serialization_and_tool_replay():
    state = RunState(
        incident=INCIDENT, investigation_start=START, investigation_end=END,
        limits=RunLimits(), allowed_tools=("query_metrics",),
    )
    original = RunHarness(state)
    original.begin_finalization(StopReason.REPEATED_TOOL_CALL)
    restored = RunState.model_validate_json(state.model_dump_json())
    resumed = RunHarness(restored, resume=True)
    assert resumed.finalization_reason is StopReason.REPEATED_TOOL_CALL
    with pytest.raises(InvestigationClosed):
        resumed.authorize_tool("query_metrics")
    resumed.begin_finalization(StopReason.NO_PROGRESS)
    phases = [e for e in restored.events if e.kind == "policy.finalizing"]
    assert len(phases) == 1
    assert resumed.finalization_reason is StopReason.REPEATED_TOOL_CALL
    # Closure never changes the trusted allowlist into permission to call more.
    with pytest.raises(RunStopped) as caught:
        resumed.authorize_tool("unregistered_tool")
    assert caught.value.reason is StopReason.TOOL_DENIED


def test_finalization_preserves_context_gate(tmp_path):
    def respond(messages, calls):
        # Grow non-reducible assistant content after the first admitted call.
        response = tool_call()
        response.content = "x" * 40_000
        return response

    model = ScriptedModel(respond=respond)
    run = run_service(make_service(
        model, Telemetry(), tmp_path, limits=RunLimits(tool_calls=1),
    ))
    assert run.state.stop_reason is StopReason.CONTEXT_LIMIT
    assert model.calls == 1
    assert any(e.kind == "policy.finalizing" for e in run.state.events)


@pytest.mark.parametrize("limits", [
    RunLimits(tool_calls=1), RunLimits(model_calls=3),
])
def test_deepseek_payload_disables_tools_and_repairs_unwanted_call(
    tmp_path, limits,
):
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        pending = set()
        for message in payload["messages"]:
            if message["role"] == "tool":
                assert message["tool_call_id"] in pending
                pending.remove(message["tool_call_id"])
            else:
                assert not pending
                pending.update(c["id"] for c in message.get("tool_calls", []))
        assert not pending
        if len(requests) <= 2:
            name, args = "query_metrics", {"expression": "up"}
        else:
            observation = next(m for m in payload["messages"]
                               if m.get("tool_call_id") == "call-1")
            name, args = "DiagnosisProposal", {
                "status": "complete", "root_cause": "Payment Charge failed",
                "confidence": .8,
                "evidence_ids": [json.loads(observation["content"])[0]["id"]],
            }
        if len(requests) > 1:
            assert payload["messages"][-1]["role"] == "user"
            assert "Investigation is now closed" in (
                payload["messages"][-1]["content"])
            assert [t["function"]["name"] for t in payload["tools"]] == [
                "DiagnosisProposal",
            ]
            assert payload["tool_choice"] == "required"
        return httpx.Response(200, json={
            "id": f"completion-{len(requests)}", "object": "chat.completion",
            "created": 0, "model": "deepseek-flash",
            "choices": [{"index": 0, "finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": "",
                                     "tool_calls": [{
                                         "id": f"call-{len(requests)}",
                                         "type": "function", "function": {
                                             "name": name,
                                             "arguments": json.dumps(
                                                 proposal_payload(args)
                                                 if name == "DiagnosisProposal"
                                                 else args),
                                         },
                                     }]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                      "total_tokens": 15},
        })

    telemetry = Telemetry()

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond)
        ) as client:
            model = BaseChatOpenAI(
                model="deepseek-flash", api_key="test",
                base_url="https://api.deepseek.com", http_async_client=client,
                profile={"tool_calling": True, "structured_output": False},
                max_retries=0,
            )
            return await make_service(
                model, telemetry, tmp_path, limits=limits,
            ).diagnose_with_evidence(INCIDENT, start=START, end=END)

    run = asyncio.run(scenario())
    assert run.state.status is RunStatus.SUCCEEDED
    assert run.state.diagnosis_repairs == 1
    assert run.state.model_calls == len(requests) == 3
    assert run.state.tool_calls == {"query_metrics": 1}
    assert len(telemetry.calls) == 1
    assert run.state.usage.reported_total_tokens == 45
