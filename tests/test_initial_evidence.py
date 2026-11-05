"""Exercise initial lookups through real binding, budgets and persistence."""

import asyncio
from datetime import UTC, datetime, timedelta
import json

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from test_harness import ScriptedModel, tool_call
from tool_fixtures import FixtureToolClient

from tracemind.agent.errors import DiagnosisCancelled
from tracemind.agent.runtime.harness import RunHarness
from tracemind.agent.runtime.runner import LangChainAgentRunner
from tracemind.agent.runtime.tools import EvidenceCollector, create_mcp_tools
from tracemind.agent.service import DiagnosisService
from tracemind.agent.verification import review_messages, subject_for
from tracemind.agent.workflow.intake import collect_initial_evidence
from tracemind.agent.workflow.context import _evidence_preview, intake_context
from tracemind.models import (
    DiagnosisStatus, Evidence, EvidenceSource, Incident,
)
from tracemind.models.intake import InitialContext
from tracemind.models.result import DiagnosisDraft
from tracemind.models.run import RunLimits, RunPurpose, RunState
from tracemind.telemetry.errors import RemoteTelemetryError


START = datetime(2024, 9, 1, tzinfo=UTC)
END = START + timedelta(minutes=10)
TRACE = "ab" * 16
SPAN = "cd" * 8


def observation(**updates):
    values = {
        "source": EvidenceSource.TRACE, "service": "payment",
        "observed_at": START + timedelta(seconds=10),
        "summary": "Payment charge failed",
        "attributes": {
            "trace_id": TRACE, "span_id": SPAN, "operation": "Charge",
            "resource": {
                "deployment.environment.name": "production",
                "service.version": "v2", "service.instance.id": "pod-a",
            },
            "code": {"file": "charge.js", "line": 48},
            "exceptions": [{
                "type": "Error", "message": "charge rejected",
                "stacktrace": "stack\n" * 2000,
                "timestamp": (START + timedelta(seconds=11)).isoformat(),
            }],
        },
    }
    values.update(updates)
    return Evidence(**values)


class Backend:
    def __init__(self, items=None, error=None):
        self.items = [observation()] if items is None else items
        self.error = error
        self.calls = []
        self.before_read = None

    async def get_trace_detail(self, trace_id, **kwargs):
        if self.before_read is not None:
            self.before_read()
        self.calls.append((trace_id, kwargs))
        if self.error is not None:
            raise self.error
        return self.items


class InspectRunner:
    def __init__(self, check=None):
        self.check = check
        self.prompts = []

    async def run(self, prompt, tools, *, recursion_limit, harness):
        self.prompts.append(prompt)
        if self.check is not None:
            self.check(prompt, harness)
        return DiagnosisDraft(
            status=DiagnosisStatus.INCONCLUSIVE,
            root_cause="Retained observations need further confirmation",
            confidence=0,
            evidence_ids=harness.state.evidence_ids,
        )


def context(**updates):
    return InitialContext(**{"traces": [{"trace_id": TRACE}], **updates})


def service(backend, runner=None, **kwargs):
    return DiagnosisService(
        runner or InspectRunner(),
        tool_client=FixtureToolClient(None, backend, None),
        allowed_tools=kwargs.pop(
            "allowed_tools", ("get_trace_detail", "read_evidence"),
        ),
        **kwargs,
    )


def run(backend, supplied=None, runner=None, **kwargs):
    agent = service(backend, runner, **kwargs)
    return asyncio.run(agent.diagnose_with_evidence(
        Incident(title="checkout", description="failure"),
        start=START, end=END, initial_context=supplied,
    ))


def packet(prompt):
    return json.loads(prompt.split("gaps using authorized tools.\n", 1)[1])


def test_first_model_input_contains_locators_and_retained_evidence():
    backend = Backend()

    def check(prompt, harness):
        assert backend.calls == [(TRACE, {})]
        assert harness.state.tool_calls == {"get_trace_detail": 1}
        data = packet(prompt)
        assert data["reported_context_is_unverified"]
        assert data["lookup_results"][0]["status"] == "found"
        preview = data["evidence_previews"][0]
        assert preview["id"] == "E1"
        assert preview["attributes"]["resource"]["service.version"] == "v2"
        assert preview["attributes"]["code"]["line"] == 48
        assert preview["attributes"]["exceptions"][0]["has_stacktrace"]
        assert preview["truncated"]

    result = run(backend, context(), InspectRunner(check))
    assert result.state.intake_progress.attempts == 1
    assert result.state.intake_progress.completed
    event = result.evidence[0].attributes["exceptions"][0]
    assert event["in_investigation_window"] is True


def test_native_agent_can_read_prefetched_evidence_on_first_model_call():
    backend = Backend()

    def respond(messages, calls):
        if calls == 1:
            assert len(backend.calls) == 1
            prompt = next(m.content for m in messages
                          if isinstance(m, HumanMessage))
            assert packet(prompt)["evidence_previews"][0]["id"] == "E1"
            return tool_call("read_evidence", {"evidence_id": "E1"})
        detail = next(m for m in reversed(messages)
                      if isinstance(m, ToolMessage)
                      and m.name == "read_evidence")
        assert "charge.js" in detail.content
        return tool_call("DiagnosisProposal", {
            "status": "inconclusive", "root_cause": "Needs release mapping",
            "confidence": 0, "evidence_ids": ["E1"], "remediation": [],
        })

    model = ScriptedModel(respond=respond)
    result = run(backend, context(), LangChainAgentRunner(model))
    assert model.calls == 2
    assert result.state.tool_calls == {
        "get_trace_detail": 1, "read_evidence": 1,
    }
    assert result.diagnosis.evidence_ids == [result.evidence[0].id]


@pytest.mark.parametrize("uppercase", [False, True])
def test_known_span_is_exact_and_service_hints_cannot_hide_it(uppercase):
    item = observation()
    if uppercase:
        item.attributes.update(trace_id=TRACE.upper(), span_id=SPAN.upper())
    backend = Backend([item])
    result = run(backend, context(traces=[{
        "trace_id": TRACE, "span_id": SPAN, "service": "checkout",
        "operation": "WrongReportedOperation",
    }]))
    assert backend.calls == [(TRACE, {"span_id": SPAN})]
    found = result.state.intake_progress.results[0]
    assert found.status == "found"
    assert found.reason == "reported_locator_hints_unconfirmed"
    assert result.evidence[0].service == "payment"


def test_initial_packet_retains_backend_pagination_and_filtered_match_count():
    item = observation()
    item.attributes["coverage"] = {
        "trace_span_count": 80, "matched_span_count": 80,
        "page_offset": 0, "page_size": 25, "next_offset": 25,
        "trace_fingerprint": "not-needed-in-initial-summary",
    }
    runner = InspectRunner()
    run(Backend([item]), context(), runner)
    result = packet(runner.prompts[0])["lookup_results"][0]
    assert result["status"] == "found"
    assert result["backend_coverage"]["next_offset"] == 25
    assert result["backend_coverage"]["page_size"] == 25
    assert result["evidence_ids"] == ["E1"]


def test_no_context_retains_zero_prefetch_behavior():
    backend = Backend()
    runner = InspectRunner()
    result = run(backend, runner=runner)
    assert backend.calls == []
    assert result.state.intake_progress is None
    assert "Initial lookup packet" not in runner.prompts[0]


def test_prefetch_toggle_keeps_hints_but_does_not_read():
    backend = Backend()
    runner = InspectRunner()
    result = run(backend, context(prefetch_traces=False), runner)
    assert backend.calls == []
    assert result.state.intake_progress.attempts == 0
    found = result.state.intake_progress.results[0]
    assert found.reason == "prefetch_disabled"
    data = packet(runner.prompts[0])
    assert data["reported_context"]["traces"][0]["trace_id"] == TRACE
    assert data["reported_context_is_unverified"]


def test_long_multibyte_hints_preserve_evidence_and_mark_reduction():
    supplied = context(
        origin={"source": "错" * 120, "external_id": "😀" * 256},
        traces=[{"trace_id": TRACE, "service": "错" * 120,
                 "operation": "😀" * 240}] * 3,
        scope={"environment": "错" * 120, "cluster": "错" * 120,
               "namespace": "错" * 120},
        error={"type": "错" * 240, "message": "😀" * 1000,
               "file": "错" * 1024},
        deployments=[{"service": "错" * 120, "version": "😀" * 240,
                      "instance": "错" * 240}] * 5,
    )
    state = RunState(
        incident=Incident(title="x", description="x"),
        initial_context=supplied, investigation_start=START,
        investigation_end=END, limits=RunLimits(), allowed_tools=(),
    )
    serialized = intake_context(state, [observation()])
    data = json.loads(serialized)
    assert len(serialized.encode()) <= (
        state.limits.context_tokens - state.limits.output_tokens
    ) // 3
    assert len(data["evidence_previews"]) == 1
    assert "origin.source" in data["truncated_reported_fields"]
    assert data["reported_context"]["traces"][0]["trace_id"] == TRACE
    assert state.initial_context == supplied


def test_preview_retains_conflicts_and_exception_only_location():
    item = observation()
    item.attributes["resource_metadata"] = {
        "conflicts": {"service.version": [
            {"source": "process", "value": "v2"},
            {"source": "span", "value": "v1"},
        ]},
    }
    item.attributes.pop("code")
    item.attributes["exceptions"][0]["code"] = {
        "file": "charge.js", "line": 48,
    }
    preview = _evidence_preview(item, max_bytes=1200)
    resource = preview["attributes"]["resource"]
    assert "service.version" not in resource
    assert resource["conflicting_fields"] == ["service.version"]
    assert preview["attributes"]["exceptions"][0]["code"]["line"] == 48
    assert len(json.dumps(preview, ensure_ascii=False,
                          separators=(",", ":")).encode()) <= 1200


@pytest.mark.parametrize("prior_marker,new_end,expected", [
    (True, START + timedelta(minutes=1), False),
    (False, END, True),
])
def test_persisted_evidence_recomputes_exception_window(
    prior_marker, new_end, expected,
):
    item = observation()
    event = item.attributes["exceptions"][0]
    event["timestamp"] = (START + timedelta(minutes=3)).isoformat()
    event["in_investigation_window"] = prior_marker
    state = RunState(
        incident=Incident(title="x", description="x"),
        investigation_start=START, investigation_end=new_end,
        limits=RunLimits(), allowed_tools=("read_evidence",),
        purpose=RunPurpose.INVESTIGATE,
    )
    result = asyncio.run(service(Backend()).run_persisted(
        state, [item], persist=lambda *_: None, is_cancelled=lambda: False,
        resume=False, graph_resume=False,
    ))
    current = result.evidence[0].attributes["exceptions"][0]
    assert current["in_investigation_window"] is expected
    assert current["timestamp"] == event["timestamp"]
    assert event["in_investigation_window"] is prior_marker


@pytest.mark.parametrize("case,status", [
    ("404", "not_found"), ("empty", "coverage_incomplete"),
    ("old", "outside_window"), ("wrong_scope", "scope_mismatch"),
    ("missing_scope", "scope_unknown"),
    ("redacted_scope", "scope_unknown"),
    ("truncated_scope", "scope_unknown"),
])
def test_missing_or_inapplicable_observations_are_explicit_gaps(case, status):
    evidence = observation()
    if case == "old":
        evidence.observed_at = START - timedelta(seconds=1)
    if case == "wrong_scope":
        evidence.attributes["resource"]["deployment.environment.name"] = (
            "staging"
        )
    if case == "missing_scope":
        evidence.attributes.pop("resource")
    if case in {"redacted_scope", "truncated_scope"}:
        field = ("redacted_fields" if case == "redacted_scope"
                 else "truncated_fields")
        evidence.attributes["resource_metadata"] = {
            field: ["deployment.environment.name"],
        }
    backend = Backend(
        items=[] if case == "empty" else [evidence],
        error=RemoteTelemetryError("not_found", status_code=404)
        if case == "404" else None,
    )
    result = run(backend, context(scope={"environment": "production"}))
    assert result.state.intake_progress.results[0].status == status
    assert result.evidence == []
    assert result.state.reference_map == {}


def test_local_attempt_cap_includes_retries_and_leaves_run_tools_open():
    backend = Backend(error=RemoteTelemetryError(
        "transient", status_code=503, retryable=True,
    ))
    result = run(backend, context(), limits=RunLimits(
        retry_attempts=5, retry_backoff_seconds=0, retry_max_backoff_seconds=0,
        tool_calls=12,
    ))
    assert len(backend.calls) == 3
    assert result.state.intake_progress.attempts == 3
    assert result.state.intake_progress.results[0].reason == (
        "intake_attempt_budget"
    )
    assert result.state.tool_calls == {"get_trace_detail": 3}
    assert not any(e.kind == "policy.finalizing" for e in result.state.events)


def test_initial_reads_cannot_exceed_smaller_global_tool_budget():
    backend = Backend(items=[])
    result = run(backend, context(traces=[
        {"trace_id": TRACE}, {"trace_id": "ef" * 16},
    ]), limits=RunLimits(tool_calls=1))
    assert len(backend.calls) == 1
    assert result.state.intake_progress.attempts == 1
    assert result.state.intake_progress.results[1].reason == (
        "investigation_tools_closed"
    )


@pytest.mark.parametrize("cancel", [False, True])
def test_initial_lookup_timeout_and_cancellation_stop_backend(cancel):
    async def scenario():
        entered = asyncio.Event()
        stopped = asyncio.Event()

        class BlockingBackend:
            async def get_trace_detail(self, *_args, **_kwargs):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()

        runner = InspectRunner()
        agent = service(BlockingBackend(), runner, limits=RunLimits(
            tool_timeout_seconds=10 if cancel else 0.02, retry_attempts=0,
        ))
        task = asyncio.create_task(agent.diagnose_with_evidence(
            Incident(title="failure", description="failure"),
            start=START, end=END, initial_context=context(),
        ))
        await asyncio.wait_for(entered.wait(), timeout=2)
        if cancel:
            task.cancel()
            with pytest.raises(DiagnosisCancelled) as caught:
                await task
            state = caught.value.state
            assert runner.prompts == []
            assert not state.intake_progress.completed
        else:
            result = await asyncio.wait_for(task, timeout=2)
            state = result.state
            assert state.intake_progress.results[0].status == "unavailable"
            assert state.intake_progress.completed
        assert stopped.is_set()
        assert state.intake_progress.attempts == 1
        assert state.tool_calls == {"get_trace_detail": 1}

    asyncio.run(scenario())


def test_disallowed_lookup_is_not_called_or_promoted_to_evidence():
    backend = Backend()
    result = run(backend, context(), allowed_tools=("read_evidence",))
    assert backend.calls == []
    assert result.state.intake_progress.results[0].reason == (
        "tool_not_authorized"
    )
    assert result.state.tool_calls == {}


def test_version_hint_does_not_filter_differently_versioned_downstream():
    checkout = observation(service="checkout", attributes={
        "trace_id": TRACE, "span_id": "ef" * 8,
        "resource": {"service.version": "v4"},
    })
    result = run(Backend([checkout, observation()]), context(
        deployments=[{"service": "checkout", "version": "v4"}],
    ))
    assert {e.service for e in result.evidence} == {"checkout", "payment"}
    assert result.evidence[1].attributes["resource"]["service.version"] == "v2"


def test_exception_time_is_not_replaced_with_span_start():
    evidence = observation()
    event = evidence.attributes["exceptions"][0]
    event["timestamp"] = (END + timedelta(seconds=1)).isoformat()
    result = run(Backend([evidence]), context())
    retained = result.evidence[0].attributes["exceptions"][0]
    assert retained["timestamp"] == event["timestamp"]
    assert retained["in_investigation_window"] is False
    assert "in_investigation_window" not in event


def test_completed_intake_survives_state_roundtrip_without_requery():
    async def scenario():
        backend = Backend()
        state = RunState(
            incident=Incident(title="failure", description="failure"),
            initial_context=context(), investigation_start=START,
            investigation_end=END, limits=RunLimits(),
            allowed_tools=("get_trace_detail", "read_evidence"),
        )
        saved = []
        collector = EvidenceCollector()

        async def persist(current):
            saved.append((current.model_dump_json(), [
                e.model_dump_json() for e in collector.evidence
            ]))

        def admitted_before_read():
            checkpoint, retained = saved[-1]
            admitted = RunState.model_validate_json(checkpoint)
            assert admitted.intake_progress.attempts == 1
            assert admitted.tool_calls == {"get_trace_detail": 1}
            assert retained == []

        backend.before_read = admitted_before_read

        harness = RunHarness(state, persist=persist)
        client = FixtureToolClient(None, backend, None)
        tools = await create_mcp_tools(client, start=START, end=END,
                                       collector=collector, harness=harness)
        await collect_initial_evidence(state, collector, harness, tools)
        stored, observations = saved[-1]
        restored = RunState.model_validate_json(stored)
        restored_collector = EvidenceCollector(initial=[
            Evidence.model_validate_json(e) for e in observations])
        resumed = RunHarness(restored, resume=True)
        rebound = await create_mcp_tools(
            client, start=START, end=END,
            collector=restored_collector, harness=resumed,
        )
        await collect_initial_evidence(
            restored, restored_collector, resumed, rebound,
        )
        assert len(backend.calls) == 1
        assert restored.intake_progress == state.intake_progress
        assert restored.reference_map == state.reference_map
    asyncio.run(scenario())


def test_explanation_never_executes_initial_queries():
    async def scenario():
        state = RunState(
            incident=Incident(title="failure", description="failure"),
            initial_context=context(), purpose=RunPurpose.EXPLAIN,
            investigation_start=START, investigation_end=END,
            limits=RunLimits(), allowed_tools=("read_evidence",),
        )
        await collect_initial_evidence(
            state, EvidenceCollector(), RunHarness(state), [],
        )
        assert state.intake_progress is None
        assert state.tool_calls == {}
    asyncio.run(scenario())


def test_verifier_keeps_reported_version_outside_observation_records():
    result = run(Backend(), context(deployments=[
        {"service": "payment", "version": "v99"},
    ]))
    subject = subject_for(result.state, result.diagnosis, result.evidence)
    messages = review_messages(subject, provider="deepseek")
    data = json.loads(messages[1]["content"])
    assert data["reported_context_is_unverified"]
    assert data["reported_context"]["deployments"][0]["version"] == "v99"
    resource = data["evidence"][0]["attributes"]["resource"]
    assert resource["service.version"] == "v2"
    assert "not E observations" in messages[0]["content"]
