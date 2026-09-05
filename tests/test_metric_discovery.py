"""Verify metadata, empty results and budgets through actual tool dispatch."""

import asyncio
import json
from uuid import uuid4

import httpx
from langchain_core.messages import ToolMessage
from otel_telemetry_mcp.telemetry.prometheus import PrometheusClient
import pytest
from test_harness import (
    END,
    INCIDENT,
    START,
    ScriptedModel,
    Telemetry,
    make_service,
    tool_call,
)
from tool_fixtures import bind_fixture_tools

from tracemind.agent.runtime.harness import RunHarness
from tracemind.agent.runtime.recovery import ToolUnavailable
from tracemind.agent.runtime.tools import EvidenceCollector
from tracemind.api.progress import readable_event
from tracemind.models.run import RunLimits, RunState, RunStatus, StopReason


def metadata_response(request):
    """Serve a metadata catalogue and a genuinely empty instant query."""
    if request.url.path == "/api/v1/label/__name__/values":
        return httpx.Response(200, json={"status": "success", "data": [
            "rpc_calls_total",
        ]})
    if request.url.path == "/api/v1/series":
        return httpx.Response(200, json={"status": "success", "data": [
            {"__name__": "rpc_calls_total", "job": "payment",
             "rpc_grpc_status_code": "13"},
        ]})
    assert request.url.path == "/api/v1/query"
    return httpx.Response(200, json={"status": "success", "data": {
        "resultType": "vector", "result": [],
    }})


def test_discovery_and_empty_query_do_not_create_evidence_or_progress():
    async def execute():
        state = RunState(
            incident=INCIDENT, investigation_start=START,
            investigation_end=END, limits=RunLimits(),
            allowed_tools=("discover_metrics", "query_metrics"),
        )
        harness, collector = RunHarness(state), EvidenceCollector()
        harness.check_progress()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(metadata_response),
        ) as http:
            tools = {t.name: t for t in bind_fixture_tools(
                PrometheusClient("http://prometheus", http), None, None,
                start=START, end=END, collector=collector, harness=harness,
            )}
            discovery = await tools["discover_metrics"].ainvoke({
                "service": "payment",
            })
            empty = await tools["query_metrics"].ainvoke({
                "expression": 'rpc_calls_total{job="payment"}',
            })
            with pytest.raises(ToolUnavailable) as rejected:
                await tools["discover_metrics"].ainvoke({"limit": 51})
            assert rejected.value.failure.category == "invalid_request"
        harness.check_progress()
        assert discovery[0]["metrics"][0]["metric"] == "rpc_calls_total"
        assert len(empty) == 1
        assert empty[0]["kind"] == "query_feedback"
        assert empty[0]["new_evidence"] == 0
        assert "id" not in empty[0]
        assert "does not establish health" in (
            tools["query_metrics"].description)
        assert collector.evidence == []
        assert state.evidence_ids == []
        assert state.progress_count == 0
        assert state.stalled_rounds == 1
        # Remote validation follows admission; rejected RPCs consume
        # an attempt.
        assert state.tool_calls == {"discover_metrics": 2, "query_metrics": 1}
        assert state.backend_attempts == 3
        completed = [e for e in state.events if e.kind == "tool.completed"]
        assert all(e.data["evidence_ids"] == [] for e in completed)
        assert completed[-1].data["result_count"] == 0
        started = next(e for e in state.events if e.kind == "tool.started")
        assert readable_event(started).message == (
            "Reading metric names and labels for payment"
        )

    asyncio.run(execute())


def test_real_loop_can_discover_query_and_return_grounded_uncertainty(
    tmp_path,
):
    def respond(messages, calls):
        if calls == 1:
            return tool_call("discover_metrics", {"service": "payment"})
        if calls == 2:
            result = json.loads(next(m.content for m in messages
                                     if isinstance(m, ToolMessage)))
            example = result[0]["metrics"][0]["example_labels"][0]
            assert example["rpc_grpc_status_code"] == "13"
            return tool_call("query_metrics", {
                "expression": 'rpc_calls_total{job="payment"}',
            })
        empty = json.loads(next(m.content for m in reversed(messages)
                                if isinstance(m, ToolMessage)))
        assert len(empty) == 1
        assert empty[0]["kind"] == "query_feedback"
        assert empty[0]["new_evidence"] == 0
        assert "id" not in empty[0]
        return tool_call("DiagnosisDraft", {
            "status": "inconclusive", "confidence": 0,
            "root_cause": "No matching metric samples; cause remains unknown",
            "evidence_ids": [], "remediation": ["Check metric availability"],
        })

    async def execute():
        model = ScriptedModel(respond=respond)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(metadata_response),
        ) as http:
            service = make_service(
                model, PrometheusClient("http://prometheus", http), tmp_path,
                allowed_tools=("discover_metrics", "query_metrics"),
            )
            result = await service.diagnose_with_evidence(
                INCIDENT, start=START, end=END,
            )
        assert result.state.status is RunStatus.INCONCLUSIVE
        assert result.state.stop_reason is StopReason.INSUFFICIENT_EVIDENCE
        assert result.evidence == []
        assert model.calls == 3
        assert result.tool_call_count == 2
        assert result.state.progress_count == 0

    asyncio.run(execute())


@pytest.mark.parametrize("reason, limits", [
    (StopReason.NO_PROGRESS, RunLimits(no_progress_rounds=1)),
    (StopReason.TOOL_BUDGET, RunLimits(tool_calls=1)),
])
def test_metadata_reads_trigger_existing_closure_limits(
    tmp_path, reason, limits,
):
    def respond(messages, calls):
        if "Investigation is closed" not in messages[0].text:
            return tool_call("discover_metrics", {"limit": calls})
        assert model.bindings[-1] == ["DiagnosisDraft"]
        return tool_call("DiagnosisDraft", {
            "status": "inconclusive", "confidence": 0,
            "root_cause": "Metadata alone cannot establish a root cause",
        })

    model = ScriptedModel(respond=respond)

    async def execute():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(metadata_response),
        ) as http:
            service = make_service(
                model, PrometheusClient("http://prometheus", http), tmp_path,
                allowed_tools=("discover_metrics",), limits=limits,
            )
            return await service.diagnose_with_evidence(
                INCIDENT, start=START, end=END,
            )

    result = asyncio.run(execute())
    assert result.tool_calls == {"discover_metrics": 1}
    assert result.evidence == []
    assert model.calls == 2
    phase = next(e for e in result.state.events
                 if e.kind == "policy.finalizing")
    assert phase.data["reason"] == reason.value


def test_metadata_cannot_be_used_as_citation_in_real_loop(tmp_path):
    def respond(messages, calls):
        if calls == 1:
            return tool_call("discover_metrics", {"service": "payment"})
        return tool_call("DiagnosisDraft", {
            "status": "complete", "confidence": .9,
            "root_cause": "Metadata proves a failure",
            "evidence_ids": [str(uuid4())],
        })

    async def execute():
        model = ScriptedModel(respond=respond)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(metadata_response),
        ) as http:
            service = make_service(
                model, PrometheusClient("http://prometheus", http), tmp_path,
                allowed_tools=("discover_metrics",),
            )
            return await service.diagnose_with_evidence(
                INCIDENT, start=START, end=END,
            )

    result = asyncio.run(execute())
    assert result.state.stop_reason is StopReason.INVALID_DIAGNOSIS
    assert result.diagnosis.evidence_ids == []
    assert result.evidence == []


def test_explicit_existing_allowlist_does_not_gain_discovery_permission():
    state = RunState(
        incident=INCIDENT, investigation_start=START, investigation_end=END,
        limits=RunLimits(), allowed_tools=("query_metrics", "read_evidence"),
    )
    telemetry = Telemetry()
    tools = bind_fixture_tools(
        telemetry, telemetry, telemetry, start=START, end=END,
        collector=EvidenceCollector(), harness=RunHarness(state),
    )
    assert {t.name for t in tools} == {"query_metrics", "read_evidence"}


@pytest.mark.parametrize('can_read', [True, False])
def test_omitted_evidence_guidance_respects_local_tool_permission(can_read):
    from tracemind.models import Evidence, EvidenceSource

    records = [Evidence(
        source=EvidenceSource.METRIC, observed_at=END,
        summary=f'Observation {i}',
    ) for i in range(51)]

    class ManyObservations:
        async def query(self, expression, *, at):
            return records

    async def exercise():
        allowed = ('query_metrics', 'read_evidence') if can_read else (
            'query_metrics',)
        state = RunState(
            incident=INCIDENT, investigation_start=START,
            investigation_end=END, limits=RunLimits(), allowed_tools=allowed,
        )
        collector = EvidenceCollector()
        tools = {t.name: t for t in bind_fixture_tools(
            ManyObservations(), None, None, start=START, end=END,
            collector=collector, harness=RunHarness(state),
        )}
        result = await tools['query_metrics'].ainvoke({'expression': 'up'})
        assert len(collector.evidence) == 51
        assert result[-1]['omitted_count'] == 1
        guidance = result[-1]['retrieval']
        if can_read:
            assert 'read_evidence' in guidance
            page = await tools['read_evidence'].ainvoke({'offset': 50})
            assert page[0]['id'] == str(records[-1].id)
        else:
            assert 'read_evidence' not in tools
            assert 'read_evidence' not in guidance
            assert 'unavailable' in guidance

    asyncio.run(exercise())
