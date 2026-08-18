import asyncio
import inspect
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
import httpx
from langchain_core.exceptions import ModelAPIError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from openai import APIConnectionError

from tracemind.agent import (
    DiagnosisError,
    DiagnosisService,
    LangChainAgentRunner,
)
from tracemind.agent.harness import RunHarness, RunStopped
from tracemind.agent.tools import EvidenceCollector, create_telemetry_tools
from tracemind.models import (
    DiagnosisStatus,
    Evidence,
    EvidenceSource,
    Incident,
)
from tracemind.models.run import (
    TOOL_NAMES,
    RunLimits,
    RunState,
    RunStatus,
    StopReason,
)
from tracemind.telemetry.errors import TelemetryClientError


START = datetime(2024, 9, 1, tzinfo=UTC)
END = START + timedelta(hours=1)
INCIDENT = Incident(title="checkout errors", description="HTTP 500 spike")


class ScriptedModel(BaseChatModel):
    (
        "Uses the real create_agent loop and tool dispatch, "
        "without a provider."
    )

    respond: Callable
    calls: int = 0
    inputs: list[list[BaseMessage]] = Field(default_factory=list)
    bindings: list[list[str]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-harness-test"

    def bind_tools(self, tools, **kwargs):
        self.bindings.append([
            tool.name if hasattr(tool, "name")
            else tool.get("function", tool)["name"]
            for tool in tools
        ])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("tests must use the async model path")

    async def _agenerate(
        self, messages, stop=None, run_manager=None, **kwargs
    ):
        self.calls += 1
        self.inputs.append(messages)
        response = self.respond(messages, self.calls)
        if inspect.isawaitable(response):
            response = await response
        return ChatResult(generations=[ChatGeneration(message=response)])


def tool_call(name="query_metrics", arguments=None) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{
            "name": name,
            "args": arguments or {"expression": "up"},
            "id": str(uuid4()),
        }],
        usage_metadata={
            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
        },
    )


def diagnosis(messages, *, override=None) -> AIMessage:
    observation = next(message for message in reversed(messages)
                       if isinstance(message, ToolMessage)
                       and message.name != "DiagnosisDraft")
    evidence_id = json.loads(observation.content)[0]["id"]
    fields = {
        "status": "complete", "root_cause": "Payment Charge failed",
        "confidence": 0.8, "evidence_ids": [evidence_id], "remediation": [],
    }
    fields.update(override or {})
    return tool_call("DiagnosisDraft", fields)


def normal_response(messages, calls):
    return tool_call() if calls == 1 else diagnosis(messages)


class Telemetry:
    def __init__(self, *, block=False, error=None, observed_at=END):
        self.calls = []
        self.block = block
        self.error = error
        self.observed_at = observed_at
        self.entered = asyncio.Event()
        self.cancelled = False

    async def query(self, expression, *, at=None):
        self.calls.append(expression)
        self.entered.set()
        if self.error:
            raise self.error
        if self.block:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return [Evidence(
            source=EvidenceSource.METRIC, summary="Payment errors=10",
            service="payment", observed_at=self.observed_at,
        )]

    async def search_logs(self, *args, **kwargs):
        raise AssertionError("unauthorized backend must not be called")


def make_service(model, telemetry, tmp_path, **kwargs):
    return DiagnosisService(
        telemetry, telemetry, telemetry, LangChainAgentRunner(model),
        event_directory=tmp_path, **kwargs,
    )


def run_service(service):
    return asyncio.run(
        service.diagnose_with_evidence(INCIDENT, start=START, end=END)
    )


def events_at(tmp_path):
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    return [json.loads(line) for line in files[0].read_text().splitlines()]


def test_real_loop_success_records_usage_evidence_and_terminal_event(tmp_path):
    model = ScriptedModel(respond=normal_response)
    run = run_service(make_service(model, Telemetry(), tmp_path))

    assert run.diagnosis.status is DiagnosisStatus.COMPLETE
    assert run.state.status is RunStatus.SUCCEEDED
    assert run.state.stop_reason is StopReason.COMPLETED
    assert run.state.model_calls == model.calls == 2
    assert run.tool_calls == {"query_metrics": 1}
    assert run.state.evidence_ids == run.diagnosis.evidence_ids
    assert run.state.usage.reported_total_tokens == 30
    assert run.state.usage.unreported_model_calls == 0
    assert "8 model calls" in model.inputs[0][0].text
    assert "7 model calls" in model.inputs[1][0].text
    events = events_at(tmp_path)
    assert [event["sequence"] for event in events] == list(
        range(1, len(events) + 1)
    )
    assert events[-1]["kind"] == "run.finished"
    assert events[-1]["data"]["reason"] == "completed"
    model_spans = {
        event["span_id"] for event in events
        if event["kind"] == "model.started"
    }
    tool_start = next(
        event for event in events if event["kind"] == "tool.started"
    )
    assert tool_start["parent_span_id"] in model_spans


@pytest.mark.parametrize("limits, reason, calls, tools, finalizing", [
    (RunLimits(model_calls=1), StopReason.MODEL_BUDGET, 1, 1, None),
    (RunLimits(tool_calls=1), StopReason.INVALID_DIAGNOSIS, 3, 1,
     StopReason.TOOL_BUDGET),
    (RunLimits(identical_tool_calls=2), StopReason.INVALID_DIAGNOSIS, 5, 2,
     StopReason.REPEATED_TOOL_CALL),
])
def test_loop_limits_are_enforced_before_execution(
    tmp_path, limits, reason, calls, tools, finalizing,
):
    model = ScriptedModel(respond=lambda messages, calls: tool_call())
    telemetry = Telemetry()
    run = run_service(make_service(model, telemetry, tmp_path, limits=limits))
    assert run.state.status is RunStatus.INCONCLUSIVE
    assert run.state.stop_reason is reason
    assert model.calls == calls
    assert len(telemetry.calls) == run.tool_call_count == tools
    # Identical observations share a citation even across repeated queries.
    assert len(run.evidence) == 1
    completed = [event for event in run.state.events
                 if event.kind == "tool.completed"]
    assert len(completed) == tools
    assert all(event.data["evidence_ids"] == [str(run.evidence[0].id)]
               for event in completed)
    assert run.diagnosis.confidence == 0
    phases = [e.data["reason"] for e in run.state.events
              if e.kind == "policy.finalizing"]
    assert phases == ([finalizing.value] if finalizing else [])
    if finalizing:
        # A model ignoring closed tools gets only the bounded repair allowance.
        assert model.bindings[-1] == ["DiagnosisDraft"]
        assert run.state.diagnosis_repairs == limits.diagnosis_repairs


@pytest.mark.parametrize("name", [
    "search_logs", "read_evidence", "discover_metrics", "unregistered_tool",
])
def test_hidden_and_unknown_tools_are_denied_in_real_loop(tmp_path, name):
    model = ScriptedModel(
        respond=lambda messages, calls: tool_call(name, {"service": "payment"})
    )
    telemetry = Telemetry()
    with pytest.raises(DiagnosisError) as caught:
        run_service(make_service(
            model, telemetry, tmp_path, allowed_tools=("query_metrics",)
        ))
    assert caught.value.state.stop_reason is StopReason.TOOL_DENIED
    assert caught.value.state.status is RunStatus.FAILED
    assert telemetry.calls == []
    assert all("search_logs" not in tools for tools in model.bindings)
    assert events_at(tmp_path)[-1]["data"]["reason"] == "tool_denied"


def test_executor_rechecks_policy_even_when_tool_was_previously_visible():
    async def scenario():
        state = RunState(
            incident=INCIDENT,
            investigation_start=START, investigation_end=END,
            limits=RunLimits(), allowed_tools=TOOL_NAMES,
        )
        harness = RunHarness(state)
        telemetry = Telemetry()
        tools = create_telemetry_tools(
            telemetry, telemetry, telemetry, start=START, end=END,
            collector=EvidenceCollector(), harness=harness,
        )
        state.allowed_tools = ()
        with pytest.raises(RunStopped):
            await tools[0].ainvoke({"expression": "up"})
        assert not telemetry.calls
        assert not state.tool_calls
    asyncio.run(scenario())


def test_parallel_budget_preserves_started_sibling(tmp_path, monkeypatch):
    release, closed = asyncio.Event(), asyncio.Event()
    begin = RunHarness.begin_finalization

    def mark_closed(self, reason):
        begin(self, reason)
        closed.set()

    monkeypatch.setattr(RunHarness, "begin_finalization", mark_closed)

    class PendingTelemetry(Telemetry):
        async def query(self, expression, *, at=None):
            self.calls.append(expression)
            self.entered.set()
            await release.wait()
            return [Evidence(source=EvidenceSource.METRIC,
                             summary="Payment errors=10", observed_at=END)]

    def respond(messages, calls):
        if calls == 2:
            results = [json.loads(m.content) for m in messages
                       if isinstance(m, ToolMessage)]
            item = next(r[0] for r in results if isinstance(r, list))
            return tool_call("DiagnosisDraft", {
                "status": "complete", "root_cause": "Payment Charge failed",
                "confidence": .8, "evidence_ids": [item["id"]],
            })
        first = tool_call(arguments={"expression": "first"})
        second = tool_call(arguments={"expression": "second"})
        return AIMessage(
            content="", tool_calls=first.tool_calls + second.tool_calls
        )

    telemetry = PendingTelemetry()
    model = ScriptedModel(respond=respond)
    service = make_service(
        model, telemetry, tmp_path, limits=RunLimits(tool_calls=1)
    )

    async def scenario():
        task = asyncio.create_task(service.diagnose_with_evidence(
            INCIDENT, start=START, end=END,
        ))
        await asyncio.wait_for(closed.wait(), timeout=2)
        assert not task.done()
        release.set()
        return await asyncio.wait_for(task, timeout=2)

    run = asyncio.run(scenario())
    assert run.state.stop_reason is StopReason.COMPLETED
    assert len(telemetry.calls) == 1
    assert not telemetry.cancelled
    assert run.diagnosis.evidence_ids == [run.evidence[0].id]
    events = events_at(tmp_path)
    assert any(event["kind"] == "tool.skipped" for event in events)
    assert not any(event["kind"] == "tool.cancelled" for event in events)
    assert events[-1]["kind"] == "run.finished"
    assert run.state.usage.unreported_model_calls == 1


@pytest.mark.parametrize("limits, reason", [
    (RunLimits(tool_timeout_seconds=0.02, retry_backoff_seconds=0),
     StopReason.INVALID_DIAGNOSIS),
    (RunLimits(duration_seconds=0.05), StopReason.DEADLINE),
])
def test_tool_timeout_and_run_deadline_cancel_backend(
    tmp_path, limits, reason,
):
    telemetry = Telemetry(block=True)
    model = ScriptedModel(respond=lambda messages, calls: tool_call())
    service = make_service(model, telemetry, tmp_path, limits=limits)
    if reason is StopReason.DEADLINE:
        with pytest.raises(DiagnosisError) as caught:
            run_service(service)
        state = caught.value.state
        assert state.status is RunStatus.FAILED
    else:
        run = run_service(service)
        state = run.state
        assert run.diagnosis.status is DiagnosisStatus.INCONCLUSIVE
        assert len(telemetry.calls) == 6
        assert state.backend_attempts == 6
        assert any(e.kind == "policy.finalizing" and e.data["reason"]
                   == "repeated_tool_call" for e in state.events)
    assert state.stop_reason is reason
    assert telemetry.cancelled
    assert events_at(tmp_path)[-1]["data"]["reason"] == reason.value


def test_model_timeout_records_unknown_usage(tmp_path):
    async def respond(messages, calls):
        await asyncio.Event().wait()

    model = ScriptedModel(respond=respond)
    with pytest.raises(DiagnosisError) as caught:
        run_service(make_service(
            model, Telemetry(), tmp_path,
            limits=RunLimits(
                model_timeout_seconds=0.02, retry_backoff_seconds=0
            ),
        ))
    assert caught.value.state.stop_reason is StopReason.MODEL_TIMEOUT
    assert caught.value.state.usage.unreported_model_calls == 3
    assert model.calls == 3
    assert caught.value.state.usage.reported_model_calls == 0
    assert any(
        event["kind"] == "model.failed" for event in events_at(tmp_path)
    )


@pytest.mark.parametrize("override, observed_at", [
    ({"evidence_ids": [str(uuid4())]}, END),
    ({"evidence_ids": []}, END),
    ({"root_cause": "   "}, END),
    ({}, START - timedelta(seconds=1)),
])
def test_completion_gate_rejects_unsupported_final_result(
    tmp_path, override, observed_at,
):
    model = ScriptedModel(
        respond=lambda messages, calls: (
            tool_call() if calls == 1
            else diagnosis(messages, override=override)
        )
    )
    run = run_service(make_service(
        model, Telemetry(observed_at=observed_at), tmp_path
    ))
    assert run.state.stop_reason is StopReason.INVALID_DIAGNOSIS
    assert run.diagnosis.status is DiagnosisStatus.INCONCLUSIVE
    assert run.diagnosis.evidence_ids == []
    assert len(run.evidence) == 1
    assert model.calls == 3
    assert run.state.diagnosis_repairs == 1
    assert events_at(tmp_path)[-1]["data"]["status"] == "inconclusive"


def test_backend_errors_preserve_safe_trace_without_raw_data(tmp_path):
    secret = "private-secret-from-backend"
    telemetry = Telemetry(error=TelemetryClientError(secret))
    model = ScriptedModel(respond=lambda messages, calls:
                          tool_call(arguments={"expression": secret}))
    with pytest.raises(DiagnosisError) as caught:
        run_service(make_service(model, telemetry, tmp_path))
    assert caught.value.state.stop_reason is StopReason.TOOL_ERROR
    assert secret not in str(caught.value)
    assert secret not in json.dumps(events_at(tmp_path))
    assert model.calls == 1


def test_cancellation_writes_terminal_event_and_stops_backend(tmp_path):
    async def scenario():
        telemetry = Telemetry(block=True)
        model = ScriptedModel(respond=lambda messages, calls: tool_call())
        service = make_service(model, telemetry, tmp_path)
        task = asyncio.create_task(
            service.diagnose_with_evidence(INCIDENT, start=START, end=END)
        )
        await asyncio.wait_for(telemetry.entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert telemetry.cancelled
    asyncio.run(scenario())
    assert events_at(tmp_path)[-1]["data"]["status"] == "cancelled"


def test_concurrent_runs_do_not_share_evidence_or_budgets(tmp_path):
    def respond(messages, calls):
        return (
            diagnosis(messages)
            if any(isinstance(m, ToolMessage) for m in messages)
            else tool_call()
        )

    async def scenario():
        service = make_service(
            ScriptedModel(respond=respond), Telemetry(), tmp_path
        )
        return await asyncio.gather(*[
            service.diagnose_with_evidence(INCIDENT, start=START, end=END)
            for _ in range(2)
        ])
    first, second = asyncio.run(scenario())
    assert first.state.id != second.state.id
    assert first.state.model_calls == second.state.model_calls == 2
    assert set(first.state.evidence_ids).isdisjoint(second.state.evidence_ids)
    assert len(list(tmp_path.glob("*.jsonl"))) == 2


def test_framework_schema_repairs_cannot_bypass_model_budget(tmp_path):
    model = ScriptedModel(respond=lambda messages, calls:
                          tool_call("DiagnosisDraft", {"status": "invalid"}))
    run = run_service(make_service(
        model, Telemetry(), tmp_path,
        limits=RunLimits(model_calls=2, diagnosis_repairs=2),
    ))
    assert run.state.stop_reason is StopReason.MODEL_BUDGET
    assert model.calls == 2
    assert run.tool_call_count == 0


def test_invalid_tool_arguments_are_traced_and_bounded(tmp_path):
    model = ScriptedModel(respond=lambda messages, calls:
                          tool_call(arguments={"expression": None}))
    telemetry = Telemetry()
    run = run_service(make_service(
        model, telemetry, tmp_path, limits=RunLimits(model_calls=2)
    ))
    assert run.state.stop_reason is StopReason.MODEL_BUDGET
    assert telemetry.calls == []
    assert sum(
        event.kind == "tool.rejected" for event in run.state.events
    ) == 2


def test_recursion_limit_has_explicit_terminal_result(tmp_path):
    model = ScriptedModel(respond=lambda messages, calls: tool_call())
    run = run_service(make_service(
        model, Telemetry(), tmp_path, recursion_limit=1
    ))
    assert run.state.stop_reason is StopReason.RECURSION_LIMIT
    assert run.state.status is RunStatus.INCONCLUSIVE
    assert events_at(tmp_path)[-1]["kind"] == "run.finished"


@pytest.mark.parametrize("error", [
    ModelAPIError("private-provider-error"),
    ChatGoogleGenerativeAIError("private-provider-error"),
    APIConnectionError(
        request=httpx.Request("POST", "https://provider.invalid")
    ),
    httpx.ConnectError("private-provider-error"),
])
def test_provider_failure_preserves_previous_evidence(tmp_path, error):
    def respond(messages, calls):
        if calls > 1:
            raise error
        return tool_call()

    with pytest.raises(DiagnosisError) as caught:
        run_service(make_service(
            ScriptedModel(respond=respond), Telemetry(), tmp_path
        ))
    assert caught.value.state.stop_reason is StopReason.MODEL_ERROR
    assert len(caught.value.evidence) == 1
    assert caught.value.state.evidence_ids == [caught.value.evidence[0].id]
    assert "private-provider-error" not in json.dumps(events_at(tmp_path))


@pytest.mark.parametrize("stage", ["model", "tool"])
@pytest.mark.parametrize("error_type", [AttributeError, RuntimeError])
def test_programming_errors_remain_internal_errors(
    tmp_path, stage, error_type,
):
    error = error_type("private-implementation-error")

    def respond(messages, calls):
        if stage == "model" and calls > 1:
            raise error
        return tool_call()

    telemetry = Telemetry(error=error if stage == "tool" else None)
    with pytest.raises(DiagnosisError) as caught:
        run_service(make_service(
            ScriptedModel(respond=respond), telemetry, tmp_path
        ))
    assert caught.value.state.stop_reason is StopReason.INTERNAL_ERROR
    assert caught.value.state.status is RunStatus.FAILED
    assert caught.value.__cause__ is error
    events = events_at(tmp_path)
    assert events[-1]["data"]["reason"] == "internal_error"
    assert any(
        event["data"].get("error_type") == error_type.__name__
        for event in events
    )
    assert "private-implementation-error" not in json.dumps(events)


@pytest.mark.parametrize("operation", ["write", "flush", "close"])
@pytest.mark.parametrize("backend_failed", [False, True])
def test_trace_io_failure_preserves_run_outcome(
    tmp_path, monkeypatch, caplog, operation, backend_failed,
):
    fdopen = os.fdopen
    streams = []

    class FailingTrace:
        def __init__(self, stream):
            self.stream = stream
            self.finishing = False

        def write(self, text):
            self.finishing = json.loads(text)["kind"] == "run.finished"
            if operation == "write" and self.finishing:
                raise OSError("private-disk-error")
            return self.stream.write(text)

        def flush(self):
            if operation == "flush" and self.finishing:
                raise OSError("private-disk-error")
            self.stream.flush()

        def close(self):
            self.stream.close()
            if operation == "close":
                raise OSError("private-disk-error")

    def failing_fdopen(*args, **kwargs):
        stream = FailingTrace(fdopen(*args, **kwargs))
        streams.append(stream)
        return stream

    monkeypatch.setattr(os, "fdopen", failing_fdopen)
    telemetry = Telemetry(error=TelemetryClientError("private-backend-error")
                          if backend_failed else None)
    service = make_service(
        ScriptedModel(respond=normal_response), telemetry, tmp_path
    )
    if backend_failed:
        with pytest.raises(DiagnosisError) as caught:
            run_service(service)
        state = caught.value.state
        assert state.stop_reason is StopReason.TOOL_ERROR
        assert caught.value.__cause__.reason is StopReason.TOOL_ERROR
    else:
        run = run_service(service)
        state = run.state
        assert state.status is RunStatus.SUCCEEDED
        assert state.stop_reason is StopReason.COMPLETED
    assert state.events[-1].kind == "run.finished"
    assert all(stream.stream.closed for stream in streams)
    assert str(state.id) in caplog.text
    assert f"trace {operation} failed" in caplog.text
    assert "private-disk-error" not in caplog.text


def test_trace_open_failure_warns_and_keeps_in_memory_events(
    tmp_path, monkeypatch, caplog,
):
    def cannot_create_directory(*args, **kwargs):
        raise PermissionError("private-filesystem-error")

    monkeypatch.setattr(Path, "mkdir", cannot_create_directory)
    run = run_service(make_service(
        ScriptedModel(respond=normal_response), Telemetry(), tmp_path
    ))
    assert run.state.status is RunStatus.SUCCEEDED
    assert run.state.events[-1].kind == "run.finished"
    assert "cannot open trace file" in caplog.text
    assert "private-filesystem-error" not in caplog.text


def test_large_results_compress_and_omitted_counterevidence_is_retrievable(
    tmp_path,
):
    records = [Evidence(
        source=EvidenceSource.LOG, summary=f"Payment error {i}",
        service="payment", observed_at=END,
        attributes={"details": "x" * 850},
    ) for i in range(59)]
    counterevidence = Evidence(
        source=EvidenceSource.LOG, summary="Payment succeeded",
        service="payment", observed_at=END,
        attributes={"conflicting_signal": "SUCCESS", "details": "y" * 4000},
    )
    records.append(counterevidence)

    class LargeTelemetry(Telemetry):
        async def query(self, expression, *, at=None):
            return records

    def respond(messages, calls):
        if calls == 1:
            return tool_call()
        last = next(
            m for m in reversed(messages) if isinstance(m, ToolMessage)
        )
        content = json.loads(last.content)
        if calls == 2:
            assert last.artifact is not None
            assert content[-1]["omitted_count"] == 10
            assert content[0]["truncated"] is True
            return tool_call("read_evidence", {"offset": 40})
        if calls == 3:
            assert any(x.get("id") == str(counterevidence.id) for x in content)
            return tool_call("read_evidence", {
                "evidence_id": str(counterevidence.id), "offset": 0,
            })
        assert "SUCCESS" in content[0]["content"]
        return diagnosis(messages)

    model = ScriptedModel(respond=respond)
    run = run_service(make_service(model, LargeTelemetry(), tmp_path))
    assert run.state.status is RunStatus.SUCCEEDED
    assert len(run.evidence) == 60
    assert run.diagnosis.evidence_ids == [counterevidence.id]
    assert run.tool_calls == {"query_metrics": 1, "read_evidence": 2}
    selections = [event.data for event in run.state.events
                  if event.kind == "context.selected"]
    assert any(event["reduced_messages"] > 0 for event in selections)
    assert all(event["estimated_input_after"] + event["output_reserved"]
               <= run.state.limits.context_tokens for event in selections)
    assert records[-1].attributes == run.evidence[-1].attributes
    assert "SUCCESS" not in json.dumps(events_at(tmp_path))


def test_accumulating_logs_continue_after_bounded_index_compression(tmp_path):
    records = [Evidence(
        source=EvidenceSource.LOG, summary=f"Payment failure {i}",
        service="payment", observed_at=END,
        attributes={"details": "x" * 850},
    ) for i in range(300)]
    records[-1] = records[-1].model_copy(update={
        "summary": "Payment succeeded", "attributes": {"status": "SUCCESS"},
    })

    class LargeTelemetry(Telemetry):
        async def query(self, expression, *, at=None):
            phase = int(expression[-1])
            return records[(phase - 1) * 100:phase * 100]

    def respond(messages, calls):
        if calls <= 3:
            return tool_call(arguments={"expression": f"phase_{calls}"})
        last = next(m for m in reversed(messages)
                    if isinstance(m, ToolMessage))
        content = json.loads(last.content)
        if calls == 4:
            assert any("context_omitted_count" in item
                       for m in messages if isinstance(m, ToolMessage)
                       for item in json.loads(m.content))
            return tool_call("read_evidence", {"offset": 280})
        if calls == 5:
            assert str(records[-1].id) in {p.get("id") for p in content}
            return tool_call("read_evidence", {
                "evidence_id": str(records[-1].id), "offset": 0,
            })
        assert "SUCCESS" in content[0]["content"]
        return diagnosis(messages)

    model = ScriptedModel(respond=respond)
    run = run_service(make_service(
        model, LargeTelemetry(), tmp_path, recursion_limit=50,
    ))
    assert run.state.status is RunStatus.SUCCEEDED
    assert model.calls == 6
    assert len(run.evidence) == 300
    assert run.diagnosis.evidence_ids == [records[-1].id]
    selections = [e.data for e in run.state.events
                  if e.kind == "context.selected"]
    assert any(e["omitted_previews"] > 0 for e in selections)
    assert all(e["fits"] and e["estimated_input_after"] <= e["input_allowance"]
               for e in selections)


def test_context_limit_stops_before_provider_and_keeps_terminal_event(
    tmp_path,
):
    model = ScriptedModel(respond=normal_response)
    run = run_service(make_service(
        model, Telemetry(), tmp_path,
        limits=RunLimits(context_tokens=4200, output_tokens=4096),
    ))
    assert model.calls == run.state.model_calls == 0
    assert run.state.stop_reason is StopReason.CONTEXT_LIMIT
    assert run.state.status is RunStatus.INCONCLUSIVE
    assert run.state.usage.reported_model_calls == 0
    assert run.state.usage.unreported_model_calls == 0
    assert events_at(tmp_path)[-1]["data"]["reason"] == "context_limit"


def test_observation_capacity_retains_partial_evidence_and_stops(tmp_path):
    class OversizeTelemetry(Telemetry):
        async def query(self, expression, *, at=None):
            return [
                Evidence(source=EvidenceSource.LOG, summary="small",
                         observed_at=END),
                Evidence(source=EvidenceSource.LOG, summary="large" * 1000,
                         observed_at=END),
            ]

    model = ScriptedModel(respond=normal_response)
    run = run_service(make_service(
        model, OversizeTelemetry(), tmp_path,
        limits=RunLimits(observation_bytes=1000),
    ))
    assert model.calls == 1
    assert run.state.stop_reason is StopReason.CONTEXT_LIMIT
    assert [item.summary for item in run.evidence] == ["small"]
    assert run.state.evidence_ids == [run.evidence[0].id]
    assert any(
        event.kind == "observations.limit" for event in run.state.events
    )


def test_detail_reads_consume_shared_tool_budget(tmp_path):
    def respond(messages, calls):
        if calls == 1:
            return tool_call()
        if calls == 2:
            return tool_call("read_evidence", {"offset": 0})
        return diagnosis(messages)

    model = ScriptedModel(respond=respond)
    run = run_service(make_service(
        model, Telemetry(), tmp_path, limits=RunLimits(tool_calls=2)
    ))
    assert run.state.stop_reason is StopReason.COMPLETED
    assert any(e.kind == "policy.finalizing" and e.data["reason"]
               == "tool_budget" for e in run.state.events)
    assert run.tool_calls == {"query_metrics": 1, "read_evidence": 1}
    assert len(run.evidence) == 1
