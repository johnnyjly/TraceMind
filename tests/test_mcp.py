"""Exercise real stdio processes and preserve the agent harness boundary."""

import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient
from mcp import StdioServerParameters

from tracemind.agent import DiagnosisError, DiagnosisService
from tracemind.agent.harness import RunHarness, RunStopped
from tracemind.agent.recovery import ToolUnavailable, classify_failure
from tracemind.agent.service import DiagnosisDraft, LangChainAgentRunner
from tracemind.agent.tools import EvidenceCollector, create_telemetry_tools
from tracemind.config import Settings
from tracemind.mcp.client import (
    McpTelemetryBackend, open_telemetry_backend, server_parameters,
)
from tracemind.models import Incident
from tracemind.models.run import (
    TOOL_NAMES, RunLimits, RunState, RunStatus, StopReason,
)
from tracemind.telemetry.errors import RemoteTelemetryError
from test_worker import ScriptedModel
from fixtures.mcp_server import FixtureBackend


START = datetime(2024, 9, 1, tzinfo=UTC)
END = START + timedelta(hours=1)
INCIDENT = Incident(title="payment errors", description="Checkout failed")
FIXTURE = Path(__file__).parent / "fixtures" / "mcp_server.py"


def parameters(mode=None):
    """Launch our fixed test server with no provider or backend credentials."""
    return StdioServerParameters(
        command=sys.executable, args=[str(FIXTURE)] + ([mode] if mode else []),
    )


def make_tools(backend, *, limits=None, allowed=None):
    """Bind the unchanged model-facing tools to a real MCP backend."""
    state = RunState(incident=INCIDENT, investigation_start=START,
                     investigation_end=END, limits=limits or RunLimits(),
                     allowed_tools=TOOL_NAMES)
    if allowed is not None:
        state.allowed_tools = allowed
    harness = RunHarness(state)
    collector = EvidenceCollector(max_bytes=state.limits.observation_bytes)
    tools = create_telemetry_tools(None, None, None, start=START, end=END,
                                   collector=collector, harness=harness,
                                   backend=backend)
    return {tool.name: tool for tool in tools}, harness, collector


def test_real_stdio_five_tools_keep_evidence_metadata_and_window_separate():
    async def exercise():
        backend = McpTelemetryBackend(parameters(), timeout=10)
        tools, harness, collector = make_tools(backend)
        try:
            metric, trace = await asyncio.gather(
                tools["query_metrics"].ainvoke({"expression": "up"}),
                tools["search_traces"].ainvoke({"service": "checkout"}),
            )
            pid = metric[0]["attributes"]["pid"]
            assert UUID(trace[0]["id"]) == UUID(int=2)
            detail = await tools["get_trace_detail"].ainvoke({
                "trace_id": trace[0]["attributes"]["trace_id"],
            })
            assert UUID(detail[0]["id"]) == UUID(int=3)
            logs = await tools["search_logs"].ainvoke({"service": "payment"})
            assert logs[0]["summary"] == "payment error"
            assert logs[1]["excluded_outside_window"] == 1
            metadata = await tools["discover_metrics"].ainvoke({
                "service": "payment",
            })
            assert metadata[0]["metrics"][0]["metric"] == (
                "payment_errors_total")
            assert {item.id for item in collector.evidence} == {
                UUID(int=i) for i in range(1, 5)
            }
            assert sum(harness.state.tool_calls.values()) == 5
            retained = await tools["read_evidence"].ainvoke({
                "evidence_id": str(UUID(int=1)),
            })
            assert "payment_errors_total" in retained[0]["content"]
        finally:
            await backend.aclose()
        assert backend._task.done()
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    asyncio.run(exercise())


def test_denied_call_never_starts_mcp_and_parallel_calls_share_budget():
    async def exercise():
        backend = McpTelemetryBackend(parameters(), timeout=10)
        tools, harness, _ = make_tools(backend, allowed=("read_evidence",))
        assert set(tools) == {"read_evidence"}
        retained = await tools["read_evidence"].ainvoke({})
        assert retained == [{"total_evidence": 0, "next_offset": None}]
        assert backend._task is None
        with pytest.raises(RunStopped):
            await harness.tool_call(
                "query_metrics", {},
                lambda: backend.query_metrics("up", at=END),
            )
        assert backend._task is None
        await backend.aclose()

        backend = McpTelemetryBackend(parameters(), timeout=10)
        tools, harness, collector = make_tools(
            backend, limits=RunLimits(tool_calls=1),
        )
        try:
            await asyncio.gather(
                tools["query_metrics"].ainvoke({"expression": "up"}),
                tools["search_logs"].ainvoke({"service": "payment"}),
                return_exceptions=True,
            )
            assert sum(harness.state.tool_calls.values()) == 1
            assert len(collector.evidence) == 1
            assert harness.finalization_reason is StopReason.TOOL_BUDGET
        finally:
            await backend.aclose()
    asyncio.run(exercise())


def test_remote_transient_failure_retains_retry_accounting_and_safe_feedback():
    async def exercise():
        backend = McpTelemetryBackend(parameters(), timeout=10)
        tools, harness, collector = make_tools(backend)
        try:
            result = await tools["query_metrics"].ainvoke({
                "expression": "retry",
            })
            assert result[0]["source"] == "metric"
            assert harness.state.tool_calls == {"query_metrics": 2}
            assert len(collector.evidence) == 1
            events = harness.state.model_dump_json()
            assert "remote-secret" not in events
            assert "private backend payload" not in events
            failed = next(e for e in harness.state.events
                          if e.kind == "tool.failed")
            assert failed.data["category"] == "transient"
            assert any(e.kind == "recovery.retry_scheduled"
                       for e in harness.state.events)
        finally:
            await backend.aclose()
    asyncio.run(exercise())


@pytest.mark.parametrize("cancel", [False, True])
def test_timeout_or_cancel_closes_calls_and_subprocess(cancel):
    async def exercise():
        backend = McpTelemetryBackend(parameters(), timeout=10)
        sample = await backend.query_metrics("up", at=END)
        pid = sample[0].attributes["pid"]
        tools, harness, collector = make_tools(backend, limits=RunLimits(
            tool_timeout_seconds=.05, retry_attempts=0,
        ))
        try:
            task = asyncio.create_task(tools["query_metrics"].ainvoke({
                "expression": "slow",
            }))
            if cancel:
                await asyncio.sleep(.01)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(ToolUnavailable):
                    await task
            assert not collector.evidence
            expected = "tool.cancelled" if cancel else "tool.failed"
            assert any(e.kind == expected
                       for e in harness.state.events)
        finally:
            await backend.aclose()
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    asyncio.run(exercise())


@pytest.mark.parametrize("mode,expression", [("malformed", "up"),
                                             (None, "crash")])
def test_bad_reply_or_disconnect_cannot_create_evidence(mode, expression):
    async def exercise():
        backend = McpTelemetryBackend(parameters(mode), timeout=10)
        tools, harness, collector = make_tools(backend)
        try:
            with pytest.raises(RunStopped) as caught:
                await tools["query_metrics"].ainvoke({
                    "expression": expression,
                })
            assert caught.value.reason is StopReason.TOOL_ERROR
            assert not collector.evidence
            assert harness.state.tool_calls == {"query_metrics": 1}
        finally:
            await backend.aclose()
    asyncio.run(exercise())


def test_production_server_rejects_unknown_tools_and_invalid_windows():
    async def exercise():
        settings = Settings(_env_file=None, google_api_key="not-forwarded")
        params = server_parameters(settings)
        assert not any("KEY" in key for key in params.env)
        backend = McpTelemetryBackend(params, timeout=10)
        try:
            for name, args in [
                ("restart_service", {}),
                ("search_logs", {"service": "payment",
                                 "start": END.isoformat(),
                                 "end": START.isoformat()}),
            ]:
                with pytest.raises(RemoteTelemetryError) as caught:
                    await backend._call(name, args)
                assert classify_failure(caught.value).category == (
                    "invalid_request")
        finally:
            await backend.aclose()
    asyncio.run(exercise())


def test_configured_api_worker_and_evaluation_use_the_mcp_path(
    tmp_path, monkeypatch,
):
    from tracemind.agent.persistence import RunRepository
    from tracemind.agent.worker import RunWorker
    from tracemind.api.app import app, get_agent_runner, get_request_settings
    from tracemind.evaluation.models import EvaluationCase, GroundTruth
    from tracemind.evaluation.runner import run_live_evaluation
    from tracemind.evaluation.summary import build_batch_report
    from tracemind.models import EvidenceSource

    settings = Settings(
        _env_file=None, telemetry_transport="mcp",
        agent_run_database=tmp_path / "runs.db",
        agent_checkpoint_database=tmp_path / "checkpoints.db",
        agent_trace_directory=None,
    )
    monkeypatch.setattr("tracemind.agent.worker.create_chat_model",
                        lambda settings: ScriptedModel())
    monkeypatch.setattr("tracemind.evaluation.runner.create_chat_model",
                        lambda settings: ScriptedModel())
    app.dependency_overrides[get_request_settings] = lambda: settings
    app.dependency_overrides[get_agent_runner] = lambda: LangChainAgentRunner(
        ScriptedModel())

    async def exercise():
        from test_mcp_http import http_server

        async with http_server() as url:
            settings.telemetry_mcp_url = url
            try:
                async with AsyncClient(transport=ASGITransport(app=app),
                                       base_url="http://test") as client:
                    response = await client.post("/api/v1/diagnoses", json={
                        "incident": INCIDENT.model_dump(mode="json"),
                        "investigation_start": START.isoformat(),
                        "investigation_end": END.isoformat(),
                    })
                    assert response.status_code == 200
                    assert response.json()["evidence_ids"] == [
                        str(UUID(int=1))]
                repository = RunRepository(settings.agent_run_database)
                state = RunState(incident=INCIDENT, investigation_start=START,
                                 investigation_end=END, limits=RunLimits(),
                                 allowed_tools=TOOL_NAMES,
                                 status=RunStatus.QUEUED)
                repository.create(state, request_hash="mcp-worker")
                worker = RunWorker(settings, repository=repository)
                assert await worker.run_once()
                stored = repository.get(state.id)
                assert stored.state.status is RunStatus.SUCCEEDED
                assert stored.state.telemetry_transport == "mcp"
                assert stored.state.model_calls == 2
                case = EvaluationCase(id="mcp", description="MCP integration",
                                      incident=INCIDENT,
                                      ground_truth=GroundTruth(
                                          root_cause_service="payment",
                                          root_cause_terms=["payment"],
                                          required_evidence_source=(
                                              EvidenceSource.METRIC),
                                      ))
                report = await run_live_evaluation(case, settings,
                                                   start=START, end=END)
                assert report.telemetry_transport == "mcp"
                assert report.run_state.model_calls == 2
                assert report.result.passed
                baseline = report.model_copy(update={
                    "telemetry_transport": "direct",
                })
                grouped = build_batch_report([baseline, report])
                assert len(grouped.groups) == 2
                assert all(group.runs == 1
                           for group in grouped.groups.values())
            finally:
                app.dependency_overrides.clear()
    asyncio.run(exercise())


def test_direct_transport_is_lazy_and_legacy_states_default_to_direct():
    async def exercise():
        async with open_telemetry_backend(Settings(
            _env_file=None, telemetry_transport="direct",
        )) as backend:
            assert backend is None
        state = RunState(incident=INCIDENT, investigation_start=START,
                         investigation_end=END, limits=RunLimits(),
                         allowed_tools=TOOL_NAMES)
        assert state.telemetry_transport == "direct"
        async with open_telemetry_backend(Settings(
            _env_file=None, telemetry_transport="mcp",
        )) as backend:
            assert backend._task is None
        assert backend._closed
    asyncio.run(exercise())


def test_direct_and_mcp_preserve_tool_schemas_and_normalized_observations():
    async def exercise():
        direct = FixtureBackend()
        backend = McpTelemetryBackend(parameters(), timeout=10)
        direct_tools, _, direct_store = make_tools(direct)
        mcp_tools, _, mcp_store = make_tools(backend)
        try:
            for name, arguments in [
                ("query_metrics", {"expression": "up"}),
                ("discover_metrics", {"service": "payment"}),
                ("search_traces", {"service": "checkout"}),
                ("get_trace_detail", {"trace_id": "0123456789abcdef"}),
                ("search_logs", {"service": "payment", "text": "error"}),
            ]:
                assert direct_tools[name].description == (
                    mcp_tools[name].description)
                assert direct_tools[name].args_schema.model_json_schema() == (
                    mcp_tools[name].args_schema.model_json_schema())
                left = await direct_tools[name].ainvoke(arguments)
                right = await mcp_tools[name].ainvoke(arguments)
                if name == "query_metrics":
                    left[0]["attributes"].pop("pid")
                    right[0]["attributes"].pop("pid")
                assert left == right
            assert [item.id for item in direct_store.evidence] == [
                item.id for item in mcp_store.evidence
            ]
        finally:
            await backend.aclose()
    asyncio.run(exercise())


def test_mcp_observations_still_require_valid_final_citations():
    class InvalidRunner:
        async def run(self, prompt, tools, **kwargs):
            query = next(t for t in tools if t.name == "query_metrics")
            await query.ainvoke({"expression": "up"})
            return DiagnosisDraft(
                status="complete", root_cause="Payment failed",
                confidence=.8, evidence_ids=[UUID(int=999)],
            )

    async def exercise():
        backend = McpTelemetryBackend(parameters(), timeout=10)
        service = DiagnosisService(None, None, None, InvalidRunner(),
                                   telemetry_backend=backend)
        try:
            with pytest.raises(DiagnosisError) as caught:
                await service.diagnose_with_evidence(
                    INCIDENT, start=START, end=END,
                )
            assert caught.value.state.stop_reason is (
                StopReason.INVALID_DIAGNOSIS)
            assert caught.value.state.telemetry_transport == "mcp"
            assert [item.id for item in caught.value.evidence] == [UUID(int=1)]
        finally:
            await backend.aclose()
    asyncio.run(exercise())


def test_production_prometheus_decoder_roundtrips_through_stdio():
    async def exercise():
        backend = McpTelemetryBackend(parameters("http"), timeout=10)
        service = DiagnosisService(None, None, None,
                                   LangChainAgentRunner(ScriptedModel()),
                                   telemetry_backend=backend)
        try:
            run = await service.diagnose_with_evidence(
                INCIDENT, start=START, end=END,
            )
            sample = run.evidence[0]
            assert run.diagnosis.evidence_ids == [sample.id]
            assert sample.summary == "Metric payment_errors_total=42"
            assert sample.service == "payment"
            assert sample.observed_at == END
            assert sample.query == "up"
            assert sample.attributes["labels"]["service_name"] == "payment"
        finally:
            await backend.aclose()
    asyncio.run(exercise())


@pytest.mark.parametrize("cancel", [False, True])
def test_interrupted_startup_closes_the_owner_task(cancel):

    async def exercise():
        backend = McpTelemetryBackend(parameters("startup"))
        try:
            tools, _, _ = make_tools(backend, limits=RunLimits(
                tool_timeout_seconds=.05, retry_attempts=0,
            ))
            task = asyncio.create_task(tools["query_metrics"].ainvoke({
                "expression": "up",
            }))
            if cancel:
                await asyncio.sleep(.01)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(ToolUnavailable):
                    await task
        finally:
            await backend.aclose()
        assert backend._closed and backend._task.done()
    asyncio.run(exercise())


def test_reply_limit_is_separate_from_partial_evidence_retention():
    async def exercise():
        params = parameters()
        params.env = {"TRACEMIND_TELEMETRY_MCP_RESPONSE_BYTES": "1024"}
        backend = McpTelemetryBackend(params, timeout=10)
        try:
            with pytest.raises(RemoteTelemetryError) as caught:
                await backend.query_metrics("oversize", at=END)
            assert caught.value.category == "response_limit"
            tools, _, collector = make_tools(backend, limits=RunLimits(
                observation_bytes=300,
            ))
            with pytest.raises(RunStopped) as stopped:
                await tools["search_logs"].ainvoke({
                    "service": "payment", "text": "capacity",
                })
            assert stopped.value.reason is StopReason.CONTEXT_LIMIT
            assert [item.id for item in collector.evidence] == [UUID(int=4)]
        finally:
            await backend.aclose()
    asyncio.run(exercise())
