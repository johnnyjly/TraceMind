"""Exercise discovered tools over real Streamable HTTP and run controls."""

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import socket
import sys
from uuid import UUID

import httpx
from otel_telemetry_mcp.config import Settings as ServerSettings
from otel_telemetry_mcp.descriptions import DESCRIPTIONS
from otel_telemetry_mcp.observations import Evidence
from otel_telemetry_mcp.server import create_app
from otel_telemetry_mcp.telemetry.errors import TelemetryClientError
from otel_telemetry_mcp.telemetry.opensearch import OpenSearchClient
import pytest
from test_worker import ScriptedModel
import uvicorn

from tracemind.agent.runtime.harness import RunHarness, RunStopped
from tracemind.agent.runtime.recovery import ToolUnavailable
from tracemind.agent.runtime.runner import LangChainAgentRunner
from tracemind.agent.service import DiagnosisService
from tracemind.agent.runtime.tools import EvidenceCollector, create_mcp_tools
from tracemind.config import Settings
from tracemind.mcp.client import HttpMcpClient, open_telemetry_backend
from tracemind.mcp.results import telemetry_reply
from tracemind.models import Incident
from tracemind.models.run import (
    RunLimits,
    RunState,
    RunStatus,
    StopReason,
    TOOL_NAMES,
)


sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "services/otel-telemetry-mcp/src"))


START = datetime(2024, 9, 1, tzinfo=UTC)
END = START + timedelta(hours=1)
INCIDENT = Incident(title="payment errors", description="Checkout failed")


class Backend:
    """Record calls and return independent service-owned observation types."""

    def __init__(self):
        self.calls = []

    async def query_metrics(self, expression, *, at):
        self.calls.append(("query_metrics", expression, at))
        if expression == "slow":
            await asyncio.sleep(.2)
        if expression == "oversize":
            return [Evidence(source="metric", summary="x" * 5000,
                             observed_at=at)]
        if expression == "retry" and len(self.calls) == 1:
            request = httpx.Request("GET", "http://backend")
            raise TelemetryClientError("private-backend-body") from (
                httpx.HTTPStatusError("private-backend-body", request=request,
                                      response=httpx.Response(
                                          503, request=request)))
        if expression == "invalid":
            request = httpx.Request("GET", "http://backend")
            raise TelemetryClientError("private-backend-body") from (
                httpx.HTTPStatusError("private-backend-body", request=request,
                                      response=httpx.Response(
                                          400, request=request)))
        return [Evidence(id=UUID(int=1), source="metric", service="payment",
                         summary="payment errors", observed_at=at)]

    async def discover_metrics(self, **arguments):
        self.calls.append(("discover_metrics", arguments))
        return {"metrics": [{"metric": "payment_errors_total"}]}

    async def search_traces(self, service, *, start, end):
        self.calls.append(("search_traces", service, start, end))
        return [Evidence(id=UUID(int=2), source="trace", service=service,
                         summary="payment Charge failed", observed_at=end,
                         attributes={"trace_id": "a" * 32})]

    async def get_trace_detail(self, trace_id):
        self.calls.append(("get_trace_detail", trace_id))
        return [Evidence(id=UUID(int=3), source="trace", service="payment",
                         summary="error span", observed_at=END)]

    async def search_logs(self, service, *, start, end, text=None):
        self.calls.append(("search_logs", service, start, end, text))
        return [Evidence(id=UUID(int=4), source="log", service=service,
                         summary="payment error", observed_at=end),
                Evidence(id=UUID(int=5), source="log", service=service,
                         summary="historical error", observed_at=start
                         - timedelta(seconds=1))]


@asynccontextmanager
async def http_server(backend=None, settings=None):
    """Run an actual SDK ASGI server on an OS-selected localhost port."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(
            create_app(settings or ServerSettings(), backend or Backend()),
            log_level="error", lifespan="on",
        ))
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(10):
                while not server.started:
                    if task.done():
                        await task
                    await asyncio.sleep(.01)
            yield f"http://127.0.0.1:{port}/mcp"
        finally:
            server.should_exit = True
            await task


def harness(allowed=TOOL_NAMES, **limits):
    """Create an isolated run without model calls or filesystem tracing."""
    return RunHarness(RunState(
        incident=INCIDENT, investigation_start=START, investigation_end=END,
        limits=RunLimits(**limits), allowed_tools=allowed,
    ))


def test_discovered_definitions_five_calls_and_local_evidence(monkeypatch):
    async def exercise():
        backend = Backend()
        description = ("Server changed guidance. "
                       + DESCRIPTIONS["query_metrics"])
        monkeypatch.setitem(DESCRIPTIONS, "query_metrics", description)
        async with http_server(backend) as url:
            client = HttpMcpClient(url)
            h = harness()
            store = EvidenceCollector()
            try:
                tools = {t.name: t for t in await create_mcp_tools(
                    client, start=START, end=END, collector=store, harness=h)}
                assert set(tools) == set(TOOL_NAMES)
                bound_description = tools["query_metrics"].description
                assert bound_description.startswith(description)
                assert "TraceMind supplies at" in bound_description
                assert "do not provide them" in bound_description
                assert "TraceMind supplies start, end" in (
                    tools["search_logs"].description)
                assert tools["get_trace_detail"].description == (
                    DESCRIPTIONS["get_trace_detail"])
                remote = {t.name: t for t in await client.discover_tools()}
                assert remote["query_metrics"].description == description
                assert remote["query_metrics"].inputSchema["required"] == [
                    "expression", "at"]
                assert remote["discover_metrics"].inputSchema["required"] == [
                    "start", "end"]
                schema = tools["query_metrics"].args_schema
                assert set(schema["properties"]) == {"expression"}
                assert schema["required"] == ["expression"]
                metric, trace = await asyncio.gather(
                    tools["query_metrics"].ainvoke({"expression": "up"}),
                    tools["search_traces"].ainvoke({"service": "checkout"}))
                await tools["get_trace_detail"].ainvoke({
                    "trace_id": trace[0]["attributes"]["trace_id"]})
                logs = await tools["search_logs"].ainvoke(
                    {"service": "payment"})
                metadata = await tools["discover_metrics"].ainvoke({})
                assert metadata[0]["metrics"][0]["metric"] == (
                    "payment_errors_total")
                assert logs[1]["excluded_outside_window"] == 1
                assert {e.id for e in store.evidence} == {
                    UUID(int=i) for i in range(1, 5)}
                metric_call = next(call for call in backend.calls
                                   if call[0] == "query_metrics")
                assert metric_call[2] == END
                assert sum(h.state.tool_calls.values()) == 5
                assert h.state.model_calls == 0
                page = await tools["read_evidence"].ainvoke({
                    "evidence_id": metric[0]["id"]})
                assert "payment errors" in page[0]["content"]
                assert len(backend.calls) == 5
                assert [e.kind for e in h.state.events if
                        e.kind.startswith("mcp.discovery")] == [
                            "mcp.discovery.started", "mcp.discovery.completed"]
            finally:
                await client.aclose()
            assert client._task.done()
    asyncio.run(exercise())


def test_local_only_never_connects_and_remote_visibility_stays_authorized():
    async def exercise():
        client = HttpMcpClient("http://127.0.0.1:1/mcp")
        h = harness(("read_evidence",))
        tools = await create_mcp_tools(client, start=START, end=END,
                                       collector=EvidenceCollector(),
                                       harness=h)
        assert [t.name for t in tools] == ["read_evidence"]
        assert client._task is None
        await client.aclose()
        async with http_server() as url:
            client = HttpMcpClient(url)
            h = harness(("search_logs", "read_evidence"))
            try:
                tools = await create_mcp_tools(client, start=START, end=END,
                                               collector=EvidenceCollector(),
                                               harness=h)
                assert {t.name for t in tools} == {
                    "search_logs", "read_evidence"}
            finally:
                await client.aclose()
    asyncio.run(exercise())


def test_http_log_query_keeps_blank_and_nonblank_records():
    class LogBackend(Backend):
        """Exercise log normalization before SDK response serialization."""

        async def search_logs(self, service, *, start, end, text=None):
            def handler(request):
                body = json.loads(request.content)
                hits = []
                if "@timestamp" in body["sort"][0]:
                    hits = [{"_id": str(i), "_source": {
                        "@timestamp": end.isoformat(), "body": value,
                        "traceId": "a" * 32,
                    }} for i, value in enumerate(
                        ("", " \t", "request failed"))]
                return httpx.Response(200, json={"hits": {"hits": hits}})

            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
            ) as http:
                return await OpenSearchClient("http://backend", "logs", http)\
                    .search_logs(service, start=start, end=end, text=text)

    async def exercise():
        async with http_server(LogBackend()) as url:
            client = HttpMcpClient(url)
            try:
                reply = await client.call("search_logs", {
                    "service": "orders", "start": START.isoformat(),
                    "end": END.isoformat(),
                })
                items = telemetry_reply(reply).evidence
                assert len(items) == 3
                assert [item.summary for item in items] == [
                    "[Empty log body]", "[Empty log body]", "request failed"]
                assert [item.attributes["original_body"]
                        for item in items[:2]] == ["", " \t"]
                assert all(item.attributes["body_empty"] for item in items[:2])
                assert "body_empty" not in items[2].attributes
                assert all(item.attributes["trace_id"] == "a" * 32
                           and item.service == "orders"
                           and item.observed_at == END for item in items)
            finally:
                await client.aclose()

    asyncio.run(exercise())


def test_http_retry_counts_attempts_and_cannot_override_window():
    async def exercise():
        backend = Backend()
        async with http_server(backend) as url:
            client = HttpMcpClient(url)
            h = harness(retry_backoff_seconds=.001)
            store = EvidenceCollector()
            try:
                tools = {t.name: t for t in await create_mcp_tools(
                    client, start=START, end=END, collector=store, harness=h)}
                result = await tools["query_metrics"].ainvoke(
                    {"expression": "retry"})
                assert result[0]["id"] == str(UUID(int=1))
                assert h.state.backend_attempts == 2
                assert h.state.tool_calls == {"query_metrics": 2}
                with pytest.raises(ToolUnavailable):
                    await tools["query_metrics"].ainvoke({
                        "expression": "up", "at": START.isoformat()})
                assert len(backend.calls) == 2
                assert "private-backend-body" not in str(h.state.events)
            finally:
                await client.aclose()
    asyncio.run(exercise())


def test_parallel_http_calls_share_budget_and_keep_prior_evidence_on_limit():
    async def exercise():
        backend = Backend()
        async with http_server(backend, ServerSettings(
            response_bytes=4096,
        )) as url:
            client = HttpMcpClient(url)
            h = harness(tool_calls=1)
            store = EvidenceCollector()
            try:
                tools = {t.name: t for t in await create_mcp_tools(
                    client, start=START, end=END, collector=store, harness=h)}
                await asyncio.gather(
                    tools["query_metrics"].ainvoke({"expression": "up"}),
                    tools["search_logs"].ainvoke({"service": "payment"}),
                    return_exceptions=True,
                )
                assert sum(h.state.tool_calls.values()) == 1
                assert len(backend.calls) == 1
                assert h.finalization_reason is StopReason.TOOL_BUDGET
            finally:
                await client.aclose()

            client = HttpMcpClient(url)
            h = harness()
            store = EvidenceCollector()
            try:
                tools = {t.name: t for t in await create_mcp_tools(
                    client, start=START, end=END, collector=store, harness=h)}
                await tools["query_metrics"].ainvoke({"expression": "up"})
                with pytest.raises(RunStopped) as error:
                    await tools["query_metrics"].ainvoke(
                        {"expression": "oversize"})
                assert error.value.reason is StopReason.TOOL_ERROR
                assert [e.id for e in store.evidence] == [UUID(int=1)]
            finally:
                await client.aclose()
    asyncio.run(exercise())


@pytest.mark.parametrize("cancel", [False, True])
def test_http_timeout_or_cancellation_closes_owner_task(cancel):
    async def exercise():
        async with http_server() as url:
            client = HttpMcpClient(url)
            h = harness(tool_timeout_seconds=.05, retry_attempts=0)
            try:
                tools = {t.name: t for t in await create_mcp_tools(
                    client, start=START, end=END,
                    collector=EvidenceCollector(), harness=h)}
                task = asyncio.create_task(tools["query_metrics"].ainvoke(
                    {"expression": "slow"}))
                if cancel:
                    await asyncio.sleep(.01)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                else:
                    with pytest.raises(ToolUnavailable):
                        await task
                assert h.state.backend_attempts == 1
            finally:
                await client.aclose()
            assert client._closed and client._task.done()
    asyncio.run(exercise())


def test_http_service_reaches_validated_diagnosis():
    async def exercise():
        async with http_server() as url:
            settings = Settings(_env_file=None, telemetry_mcp_url=url)
            async with open_telemetry_backend(settings) as client:
                service = DiagnosisService(
                    LangChainAgentRunner(ScriptedModel()),
                    tool_client=client,
                )
                run = await service.diagnose_with_evidence(
                    INCIDENT, start=START, end=END)
                assert run.state.status is RunStatus.SUCCEEDED
                assert run.state.telemetry_transport == "mcp"
                assert run.diagnosis.evidence_ids == [UUID(int=1)]
                assert run.state.model_calls == 2
    asyncio.run(exercise())


def test_unavailable_discovery_stops_before_model_and_keeps_partial_state():
    async def exercise():
        h = harness()
        client = HttpMcpClient("http://127.0.0.1:1/mcp", timeout=.1)
        try:
            with pytest.raises(RunStopped) as error:
                await create_mcp_tools(client, start=START, end=END,
                                       collector=EvidenceCollector(),
                                       harness=h)
            assert error.value.reason is StopReason.TOOL_ERROR
            assert h.state.model_calls == 0
            assert not h.state.tool_calls
            assert any(e.kind == "mcp.discovery.failed"
                       for e in h.state.events)
        finally:
            await client.aclose()
    asyncio.run(exercise())


def test_replay_tool_definitions_match_live_discovery():
    from tracemind.evaluation.replay import recorded_tools

    async def exercise():
        async with http_server() as url:
            client = HttpMcpClient(url)
            try:
                discovered = await client.discover_tools()
            finally:
                await client.aclose()

        def contract(tools):
            return {t.name: (t.description, t.inputSchema,
                             t.outputSchema, t.meta) for t in tools}
        assert contract(discovered) == contract(recorded_tools())

    asyncio.run(exercise())


def test_invalid_trace_rpc_counts_attempt_without_backend_query():
    from tracemind.agent.runtime.recovery import ToolUnavailable

    async def exercise():
        backend = Backend()
        async with http_server(backend) as url:
            client = HttpMcpClient(url)
            run = harness()
            try:
                tools = {t.name: t for t in await create_mcp_tools(
                    client, start=START, end=END,
                    collector=EvidenceCollector(), harness=run)}
                with pytest.raises(ToolUnavailable) as rejected:
                    await tools['get_trace_detail'].ainvoke(
                        {'trace_id': 'invalid'}
                    )
                assert rejected.value.failure.category == 'invalid_request'
                assert run.state.tool_calls == {'get_trace_detail': 1}
                assert backend.calls == []
            finally:
                await client.aclose()

    asyncio.run(exercise())


@pytest.mark.parametrize('entrypoint', ['api', 'worker', 'evaluation'])
def test_entrypoints_use_shared_http_service_assembly(
    tmp_path,
    monkeypatch,
    entrypoint,
):
    from tracemind.agent.worker import RunWorker
    from tracemind.api.app import (
        create_app as create_api,
        get_request_settings,
    )
    from tracemind.evaluation.models import EvaluationCase
    from tracemind.evaluation.runner import run_live_evaluation
    from tracemind.agent.storage.run_repository import RunRepository

    monkeypatch.setattr('tracemind.agent.factory.create_chat_model',
                        lambda settings: ScriptedModel())

    async def exercise():
        backend = Backend()
        async with http_server(backend) as url:
            settings = Settings(
                _env_file=None, telemetry_mcp_url=url,
                agent_run_database=tmp_path / 'runs.db',
                agent_checkpoint_database=tmp_path / 'checkpoints.db',
                agent_trace_directory=tmp_path / 'traces',
            )
            if entrypoint == 'api':
                application = create_api()
                application.dependency_overrides[get_request_settings] = (
                    lambda: settings
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=application),
                    base_url='http://test',
                ) as http:
                    response = await http.post('/api/v1/diagnoses', json={
                        'incident': INCIDENT.model_dump(mode='json'),
                        'investigation_start': START.isoformat(),
                        'investigation_end': END.isoformat(),
                    })
                assert response.status_code == 200
                assert response.json()['evidence_ids'] == [str(UUID(int=1))]
            elif entrypoint == 'worker':
                repository = RunRepository(settings.agent_run_database)
                state = harness().state.model_copy(
                    update={'status': RunStatus.QUEUED}
                )
                repository.create(state, request_hash='http-worker')
                assert await RunWorker(
                    settings,
                    repository=repository).run_once()
                stored = repository.get(state.id)
                assert stored.state.status is RunStatus.SUCCEEDED
                assert stored.state.telemetry_transport == 'mcp'
                assert stored.result.diagnosis.evidence_ids == [UUID(int=1)]
            else:
                case = EvaluationCase.model_validate_json(
                    Path('evals/cases/payment_failure.json').read_text())
                report = await run_live_evaluation(
                    case,
                    settings,
                    start=START,
                    end=END)
                assert report.error_type is None
                assert report.run_state.status is RunStatus.SUCCEEDED
                assert report.telemetry_transport == 'mcp'
            assert any(call[0] == 'query_metrics' for call in backend.calls)

    asyncio.run(exercise())
