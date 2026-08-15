"""Injected failures exercise retry admission, backoff, and safe taxonomy."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest
from langchain.agents.middleware import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.exceptions import ModelAPIError
from langchain_openai.chat_models.base import BaseChatOpenAI

from test_harness import (
    END, INCIDENT, START, ScriptedModel, Telemetry, diagnosis,
    make_service, run_service, tool_call,
)

from tracemind.agent import DiagnosisService, LangChainAgentRunner
from tracemind.agent.harness import RunHarness, RunStopped
from tracemind.agent.llm import create_chat_model, create_fallback_model
from tracemind.agent.recovery import classify_failure, retry_after_seconds
from tracemind.config import Settings
from tracemind.models import DiagnosisStatus, Incident
from tracemind.models.run import RunLimits, RunState, RunStatus, StopReason
from tracemind.telemetry.errors import TelemetryClientError


def http_failure(status, *, retry_after=None):
    """Construct a provider-like HTTP failure with private response data."""
    response = httpx.Response(
        status, request=httpx.Request("GET", "https://backend.invalid"),
        headers={"Retry-After": retry_after} if retry_after else {},
        text="private backend payload",
    )
    return httpx.HTTPStatusError(
        "private exception", request=response.request, response=response
    )


def harness(**limits):
    """Start an isolated run with configurable recovery budgets."""
    start = datetime(2024, 9, 1, tzinfo=UTC)
    return RunHarness(RunState(
        incident=Incident(title="Failure", description="HTTP 500"),
        investigation_start=start,
        investigation_end=start + timedelta(hours=1),
        limits=RunLimits(retry_backoff_seconds=0, **limits),
        allowed_tools=("query_metrics",),
    ))


@pytest.mark.parametrize("status, retryable, category", [
    (429, True, "transient"), (503, True, "transient"),
    (408, True, "transient"), (501, False, "service_error"),
    (400, False, "invalid_request"), (401, False, "access_denied"),
    (403, False, "access_denied"),
])
def test_classification_follows_http_cause(status, retryable, category):
    outer = TelemetryClientError("private wrapper")
    outer.__cause__ = http_failure(status, retry_after="2")
    failure = classify_failure(outer)
    assert failure.retryable is retryable
    assert failure.category == category
    assert failure.status_code == status
    assert failure.retry_after == (2 if retryable else None)


def test_retry_after_supports_dates_and_rejects_invalid_values():
    date = format_datetime(datetime.now(UTC) + timedelta(seconds=30))
    assert 28 <= retry_after_seconds(date) <= 30
    assert retry_after_seconds("0.5") == 0.5
    assert retry_after_seconds("-1") == 0
    for value in (None, "invalid", "inf", "nan"):
        assert retry_after_seconds(value) is None


def test_model_retries_account_attempts_and_respect_hint(monkeypatch):
    run = harness()
    calls = 0
    delays = []

    async def sleep(delay):
        delays.append(delay)

    async def call():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise http_failure(429 if calls == 1 else 503, retry_after="2")
        return ModelResponse(result=[AIMessage(
            content="done", usage_metadata={
                "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
            },
        )])

    monkeypatch.setattr("tracemind.agent.harness.asyncio.sleep", sleep)
    asyncio.run(run.model_call(call, model_name="test"))
    assert calls == run.state.model_calls == 3
    assert delays == [2, 2]
    assert run.state.usage.unreported_model_calls == 2
    assert run.state.usage.reported_model_calls == 1
    assert run.state.usage.reported_total_tokens == 15
    assert sum(e.kind == "model.failed" for e in run.state.events) == 2
    assert "private" not in run.state.model_dump_json()


def test_retry_cannot_exceed_model_budget_or_retry_after_deadline():
    for limits, reason, expected_calls in [
        ({"model_calls": 2}, StopReason.MODEL_BUDGET, 2),
        ({"duration_seconds": 1}, StopReason.DEADLINE, 1),
    ]:
        run = harness(**limits)
        calls = 0

        async def call():
            nonlocal calls
            calls += 1
            raise http_failure(503, retry_after="5" if expected_calls == 1
                               else "0")

        with pytest.raises(RunStopped) as caught:
            asyncio.run(run.model_call(call, model_name="test"))
        assert caught.value.reason is reason
        assert calls == expected_calls
        assert run.state.usage.unreported_model_calls == expected_calls


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 501])
def test_permanent_model_errors_never_retry_or_fallback(status):
    run = harness()

    async def call():
        raise http_failure(status)

    with pytest.raises(RunStopped) as caught:
        asyncio.run(run.model_call(
            call, model_name="test", allow_fallback=True
        ))
    assert caught.value.reason is StopReason.MODEL_ERROR
    assert run.state.model_calls == 1
    assert not any(
        e.kind == "recovery.retry_scheduled" for e in run.state.events
    )


def test_parallel_tool_retries_share_the_attempt_budget():
    run = harness(tool_calls=3)
    attempts = 0

    async def call():
        nonlocal attempts
        attempts += 1
        raise TelemetryClientError("wrapped") from http_failure(503)

    async def scenario():
        return await asyncio.gather(
            run.tool_call("query_metrics", {"expression": "a"}, call),
            run.tool_call("query_metrics", {"expression": "b"}, call),
            return_exceptions=True,
        )

    results = asyncio.run(scenario())
    assert attempts == run.state.backend_attempts == 3
    assert run.state.tool_calls == {"query_metrics": 3}
    assert all(isinstance(result, RunStopped) for result in results)
    assert all(result.reason is StopReason.TOOL_BUDGET for result in results)


def test_fallback_requires_explicit_pair_and_selected_provider_key():
    assert create_fallback_model(Settings(_env_file=None)) is None
    with pytest.raises(ValueError, match="set together"):
        Settings(_env_file=None, agent_fallback_provider="deepseek")
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        create_fallback_model(Settings(
            _env_file=None, agent_fallback_provider="deepseek",
            agent_fallback_model="deepseek-flash", openai_api_key="unrelated",
        ))
    model = create_fallback_model(Settings(
        _env_file=None, agent_fallback_provider="deepseek",
        agent_fallback_model="deepseek-flash", deepseek_api_key="test",
    ))
    assert model.model_name == "deepseek-flash"
    assert model.max_retries == 0


def test_tool_exhaustion_uses_remaining_source(tmp_path):
    class UnavailableTraces(Telemetry):
        trace_attempts = 0

        async def search_traces(self, *args, **kwargs):
            self.trace_attempts += 1
            raise TelemetryClientError("private") from http_failure(503)

    telemetry = UnavailableTraces()

    def respond(messages, calls):
        if calls == 1:
            return tool_call("search_traces", {"service": "checkout"})
        if calls == 2:
            assert messages[-1].status == "error"
            assert "No evidence" in messages[-1].content
            assert "private" not in messages[-1].content
            return tool_call()
        return diagnosis(messages)

    run = run_service(make_service(
        ScriptedModel(respond=respond), telemetry, tmp_path,
        limits=RunLimits(retry_backoff_seconds=0),
    ))
    assert run.diagnosis.status is DiagnosisStatus.COMPLETE
    assert telemetry.trace_attempts == 3
    assert run.tool_calls == {"search_traces": 3, "query_metrics": 1}
    assert run.state.backend_attempts == 4
    assert len(run.evidence) == 1
    assert "private" not in run.state.model_dump_json()


def test_invalid_query_is_corrected_without_transport_retry(tmp_path):
    class InvalidQuery(Telemetry):
        async def query(self, expression, *, at=None):
            if expression == "invalid":
                self.calls.append(expression)
                raise TelemetryClientError("private") from http_failure(400)
            return await super().query(expression, at=at)

    def respond(messages, calls):
        if calls < 3:
            return tool_call(arguments={
                "expression": "invalid" if calls == 1 else "up"
            })
        return diagnosis(messages)

    telemetry = InvalidQuery()
    run = run_service(make_service(ScriptedModel(respond=respond),
                                   telemetry, tmp_path))
    assert run.state.status is RunStatus.SUCCEEDED
    assert telemetry.calls == ["invalid", "up"]
    assert not any(
        e.kind == "recovery.retry_scheduled" for e in run.state.events
    )


def test_fallback_keeps_evidence_history_and_stays_selected(tmp_path):
    def primary_response(messages, calls):
        if calls > 1:
            raise ModelAPIError("private transient error")
        return tool_call()

    def fallback_response(messages, calls):
        assert any(getattr(item, "name", None) == "query_metrics"
                   for item in messages)
        if calls == 1:
            return tool_call(arguments={"expression": "second"})
        return diagnosis(messages)

    primary = ScriptedModel(respond=primary_response)
    fallback = ScriptedModel(respond=fallback_response)
    telemetry = Telemetry()
    service = DiagnosisService(
        telemetry, telemetry, telemetry,
        LangChainAgentRunner(primary, fallback_model=fallback),
        limits=RunLimits(retry_backoff_seconds=0), event_directory=tmp_path,
    )
    run = run_service(service)
    assert run.state.status is RunStatus.SUCCEEDED
    assert primary.calls == 4
    assert fallback.calls == 2
    assert run.state.model_calls == 6
    assert run.state.model_switches == 1
    assert run.state.usage.unreported_model_calls == 3
    assert run.state.usage.reported_total_tokens == 45
    assert telemetry.calls == ["up", "second"]
    assert len(run.evidence) == 1
    assert sum(e.kind == "model.switched" for e in run.state.events) == 1


def test_fallback_selection_is_run_local(tmp_path):
    def primary_response(messages, calls):
        if calls <= 3:
            raise ModelAPIError("unavailable")
        return tool_call() if calls == 4 else diagnosis(messages)

    primary = ScriptedModel(respond=primary_response)
    fallback = ScriptedModel(respond=lambda messages, calls: tool_call(
        "DiagnosisDraft", {"status": "inconclusive", "root_cause": "Unknown",
                           "confidence": 0, "evidence_ids": []}
    ))
    telemetry = Telemetry()
    service = DiagnosisService(
        telemetry, telemetry, telemetry,
        LangChainAgentRunner(primary, fallback_model=fallback),
        limits=RunLimits(retry_backoff_seconds=0), event_directory=tmp_path,
    )
    first = run_service(service)
    second = run_service(service)
    assert first.state.model_switches == 1
    assert second.state.model_switches == 0
    assert second.state.status is RunStatus.SUCCEEDED
    assert fallback.calls == 1


def test_citation_repair_reuses_graph_messages_and_accounts_usage(tmp_path):
    def respond(messages, calls):
        if calls == 1:
            return tool_call()
        if calls == 2:
            return diagnosis(messages, override={"evidence_ids": []})
        assert messages[-1].status == "error"
        return diagnosis(messages)

    run = run_service(make_service(ScriptedModel(respond=respond),
                                   Telemetry(), tmp_path))
    assert run.state.status is RunStatus.SUCCEEDED
    assert run.state.diagnosis_repairs == 1
    assert run.state.model_calls == 3
    assert run.state.usage.reported_total_tokens == 45
    assert run.tool_call_count == 1
    assert run.diagnosis.evidence_ids == [run.evidence[0].id]


@pytest.mark.parametrize("native", [True, False])
def test_malformed_schema_repair_preserves_provider_usage(tmp_path, native):
    fields = {"status": "inconclusive", "root_cause": "Missing evidence",
              "confidence": 0, "evidence_ids": []}

    def respond(messages, calls):
        if native:
            return AIMessage(
                content="invalid json" if calls == 1 else json.dumps(fields),
                usage_metadata={"input_tokens": 10, "output_tokens": 5,
                                "total_tokens": 15},
            )
        return tool_call("DiagnosisDraft", {"status": "invalid"}
                         if calls == 1 else fields)

    model = ScriptedModel(respond=respond, profile={
        "structured_output": native, "tool_calling": True,
    })
    run = run_service(make_service(model, Telemetry(), tmp_path))
    assert run.state.diagnosis_repairs == 1
    assert run.state.model_calls == 2
    assert run.state.usage.reported_total_tokens == 30
    assert run.state.usage.unreported_model_calls == 0
    assert run.state.stop_reason is StopReason.INSUFFICIENT_EVIDENCE


def test_distinct_empty_queries_stop_without_fabricating_evidence(tmp_path):
    class EmptyTelemetry(Telemetry):
        async def query(self, expression, *, at=None):
            self.calls.append(expression)
            return []

    model = ScriptedModel(respond=lambda messages, calls: tool_call(
        arguments={"expression": str(calls)}
    ))
    telemetry = EmptyTelemetry()
    run = run_service(make_service(model, telemetry, tmp_path))
    assert run.state.stop_reason is StopReason.NO_PROGRESS
    assert run.state.model_calls == 3
    assert telemetry.calls == ["1", "2", "3"]
    assert run.diagnosis.status is DiagnosisStatus.INCONCLUSIVE
    assert run.evidence == []


def test_cancel_during_backoff_prevents_another_attempt(tmp_path):
    async def scenario():
        entered = asyncio.Event()

        def respond(messages, calls):
            entered.set()
            raise ModelAPIError("unavailable")

        model = ScriptedModel(respond=respond)
        service = make_service(
            model, Telemetry(), tmp_path,
            limits=RunLimits(retry_backoff_seconds=4),
        )
        task = asyncio.create_task(service.diagnose_with_evidence(
            INCIDENT, start=START, end=END
        ))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert model.calls == 1

    asyncio.run(scenario())


def assert_tool_pairs(messages):
    """Verify each model input closes every assistant tool-call exchange."""
    pending = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            assert message.tool_call_id in pending
            pending.remove(message.tool_call_id)
        else:
            assert not pending
            if isinstance(message, AIMessage):
                pending.update(call["id"] for call in message.tool_calls)
                pending.update(c["id"] for c in message.invalid_tool_calls)
    assert not pending


@pytest.mark.parametrize("content", ["No evidence yet", ""])
def test_missing_structure_is_repaired_with_accounted_usage(tmp_path, content):
    def respond(messages, calls):
        assert_tool_pairs(messages)
        if calls == 1:
            return AIMessage(
                content=content, response_metadata={"finish_reason": "length"},
                usage_metadata={"input_tokens": 10, "output_tokens": 5,
                                "total_tokens": 15},
            )
        assert isinstance(messages[-1], HumanMessage)
        return tool_call("DiagnosisDraft", {
            "status": "inconclusive", "root_cause": "Missing evidence",
            "confidence": 0,
        })

    run = run_service(make_service(ScriptedModel(respond=respond),
                                   Telemetry(), tmp_path))
    assert run.state.stop_reason is StopReason.INSUFFICIENT_EVIDENCE
    assert run.state.diagnosis_repairs == 1
    assert run.state.model_calls == 2
    assert run.state.usage.reported_total_tokens == 30
    assert run.state.usage.reported_model_calls == 2


@pytest.mark.parametrize("model_budget, reason, calls", [
    (1, StopReason.MODEL_BUDGET, 1),
    (8, StopReason.INVALID_DIAGNOSIS, 2),
])
def test_missing_structure_repair_is_bounded(
    tmp_path, model_budget, reason, calls,
):
    model = ScriptedModel(
        respond=lambda messages, count: AIMessage(content="")
    )
    run = run_service(make_service(
        model, Telemetry(), tmp_path,
        limits=RunLimits(model_calls=model_budget),
    ))
    assert run.state.stop_reason is reason
    assert run.diagnosis.status is DiagnosisStatus.INCONCLUSIVE
    assert run.state.model_calls == model.calls == calls
    assert run.state.diagnosis_repairs == 1
    assert run.state.usage.unreported_model_calls == calls
    assert run.evidence == []


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("repairs", [0, 1, 3])
def test_repair_limits_are_independent_of_output_strategy(
    tmp_path, native, repairs,
):
    def respond(messages, calls):
        assert_tool_pairs(messages)
        fields = {
            "status": "invalid" if calls <= repairs else "inconclusive",
            "root_cause": "Missing evidence", "confidence": 0,
        }
        return (AIMessage(content=json.dumps(fields)) if native
                else tool_call("DiagnosisDraft", fields))

    model = ScriptedModel(respond=respond, profile={
        "structured_output": native, "tool_calling": True,
    })
    run = run_service(make_service(
        model, Telemetry(), tmp_path,
        limits=RunLimits(diagnosis_repairs=repairs, no_progress_rounds=1),
    ))
    assert run.state.stop_reason is StopReason.INSUFFICIENT_EVIDENCE
    assert run.state.model_calls == repairs + 1
    assert run.state.diagnosis_repairs == repairs
    assert run.state.stalled_rounds == 0


@pytest.mark.parametrize("failure", ["schema", "citation", "multiple"])
@pytest.mark.parametrize("tools_first", [True, False])
def test_mixed_rejected_diagnosis_executes_tools_and_repairs(
    tmp_path, failure, tools_first,
):
    def respond(messages, calls):
        assert_tool_pairs(messages)
        if calls == 1:
            fields = ({"status": "invalid"} if failure != "citation" else {
                "status": "complete", "root_cause": "Unsupported",
                "confidence": 1, "evidence_ids": [],
            })
            drafts = tool_call("DiagnosisDraft", fields).tool_calls
            if failure == "multiple":
                drafts += tool_call("DiagnosisDraft", fields).tool_calls
            queries = tool_call().tool_calls + tool_call(
                arguments={"expression": "second"}
            ).tool_calls
            return AIMessage(
                content="", tool_calls=(queries + drafts if tools_first
                                        else drafts + queries),
                usage_metadata={"input_tokens": 10, "output_tokens": 5,
                                "total_tokens": 15},
            )
        errors = [m for m in messages if isinstance(m, ToolMessage)
                  and m.name == "DiagnosisDraft"]
        assert len(errors) == (2 if failure == "multiple" else 1)
        assert all(m.status == "error" for m in errors)
        return diagnosis(messages)

    model, telemetry = ScriptedModel(respond=respond), Telemetry()
    run = run_service(make_service(model, telemetry, tmp_path))
    assert run.state.status is RunStatus.SUCCEEDED
    assert run.state.model_calls == model.calls == 2
    assert run.state.diagnosis_repairs == 1
    assert run.tool_calls == {"query_metrics": 2}
    assert set(telemetry.calls) == {"up", "second"}
    assert run.diagnosis.evidence_ids == [run.evidence[0].id]
    assert run.state.usage.reported_total_tokens == 30


@pytest.mark.parametrize("native", [True, False])
def test_internal_repair_history_survives_a_followup_tool_round(
    tmp_path, native,
):
    def respond(messages, calls):
        assert_tool_pairs(messages)
        if calls == 1:
            return (AIMessage(content="invalid json") if native else
                    tool_call("DiagnosisDraft", {"status": "invalid"}))
        if calls == 2:
            return tool_call()
        assert sum("Diagnosis rejected" in m.text for m in messages) == 1
        accepted = diagnosis(messages)
        return (AIMessage(content=json.dumps(accepted.tool_calls[0]["args"]))
                if native else accepted)

    model = ScriptedModel(respond=respond, profile={
        "structured_output": native, "tool_calling": True,
    })
    run = run_service(make_service(
        model, Telemetry(), tmp_path, limits=RunLimits(no_progress_rounds=1),
    ))
    assert run.state.status is RunStatus.SUCCEEDED
    assert run.state.diagnosis_repairs == 1
    assert run.state.model_calls == 3
    assert run.tool_calls == {"query_metrics": 1}


@pytest.mark.parametrize("mixed, invalid_arguments", [
    (False, False), (True, False), (False, True), (True, True),
])
def test_repair_messages_reach_deepseek_with_valid_pairs(
    tmp_path, monkeypatch, mixed, invalid_arguments,
):
    """Verify repaired graph history through the real SDK serialization."""
    requests = []

    def sdk_call(name, fields, call_id):
        return {"id": call_id, "type": "function", "function": {
            "name": name, "arguments": json.dumps(fields),
        }}

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["max_tokens"] == 512
        assert "max_completion_tokens" not in payload
        pending = set()
        for message in payload["messages"]:
            if message["role"] == "tool":
                assert message["tool_call_id"] in pending
                pending.remove(message["tool_call_id"])
            else:
                assert not pending
                pending.update(c["id"] for c in message.get("tool_calls", []))
        assert not pending
        if len(requests) == 1:
            message = {"role": "assistant", "content": "Missing structure"}
            if mixed or invalid_arguments:
                draft = sdk_call("DiagnosisDraft", {"status": "invalid"}, "d1")
                if invalid_arguments:
                    draft["function"]["arguments"] = "invalid json"
                message["tool_calls"] = [draft]
                if mixed:
                    message["tool_calls"].append(sdk_call(
                        "query_metrics", {"expression": "up"}, "q1"
                    ))
        else:
            fields = {"status": "inconclusive", "root_cause": "Unknown",
                      "confidence": 0, "evidence_ids": []}
            if mixed:
                observation = next(m for m in payload["messages"]
                                   if m.get("tool_call_id") == "q1")
                fields.update(status="complete", confidence=0.8,
                              evidence_ids=[
                                  json.loads(observation["content"])[0]["id"]
                              ])
            message = {"role": "assistant", "content": "", "tool_calls": [
                sdk_call("DiagnosisDraft", fields, "d2"),
            ]}
        return httpx.Response(200, json={
            "id": f"completion-{len(requests)}", "object": "chat.completion",
            "created": 0, "model": "deepseek-flash",
            "choices": [{"index": 0, "finish_reason": "tool_calls",
                         "message": message}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20,
                      "total_tokens": 30},
        })

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond)
        ) as http:
            monkeypatch.setattr(
                "tracemind.agent.llm.BaseChatOpenAI",
                lambda **kwargs: BaseChatOpenAI(
                    http_async_client=http, **kwargs
                ),
            )
            model = create_chat_model(Settings(
                _env_file=None, llm_provider="deepseek",
                llm_model="deepseek-flash", deepseek_api_key="test",
            ))
            return await make_service(
                model, Telemetry(), tmp_path,
                limits=RunLimits(output_tokens=512),
            ).diagnose_with_evidence(INCIDENT, start=START, end=END)

    run = asyncio.run(scenario())
    assert len(requests) == run.state.model_calls == 2
    assert run.state.diagnosis_repairs == 1
    assert run.state.usage.reported_total_tokens == 60
    assert run.state.stop_reason is (StopReason.COMPLETED if mixed else
                                     StopReason.INSUFFICIENT_EVIDENCE)
