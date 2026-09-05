"""Check open tool contracts and application-owned permission narrowing."""

import asyncio
import json

from mcp.types import CallToolResult, ImageContent, TextContent, Tool
from pydantic import ValidationError
import pytest
from test_harness import (
    END, INCIDENT, START, ScriptedModel, diagnosis, tool_call,
)

from tracemind.agent.runtime.harness import RunHarness, RunStopped
from tracemind.agent.runtime.recovery import ToolUnavailable
from tracemind.agent.runtime.runner import LangChainAgentRunner
from tracemind.agent.runtime.tools import EvidenceCollector, create_mcp_tools
from tracemind.agent.service import DiagnosisService
from tracemind.agent.storage.run_repository import RunRepository
from tracemind.agent.workflow.tool_policy import resolve_tool_policy
from tracemind.mcp.client import HttpMcpClient
from tracemind.mcp.contracts import TelemetryReply
from tracemind.mcp.results import RESULT_CONTRACT_KEY, TELEMETRY_CONTRACT
from tracemind.models import Evidence
from tracemind.models.run import (
    RunLimits, RunPurpose, RunState, RunStatus, StopReason, TOOL_NAMES,
)
from tracemind.telemetry.errors import RemoteTelemetryError


def definition(name, *, meta=None):
    """Declare an unrelated tool without any built-in domain tool name."""
    return Tool(name=name, description="Read reference information.",
                inputSchema={"type": "object", "properties": {}},
                _meta=meta)


def state(allowed, **kwargs):
    """Build a run snapshot independently of global settings."""
    return RunState(incident=INCIDENT, investigation_start=START,
                    investigation_end=END, limits=RunLimits(),
                    allowed_tools=allowed, **kwargs)


class Client:
    """Return wire results while recording admitted calls."""

    transport = "replay"

    def __init__(self, definitions, result):
        """Retain a tool catalogue and one deterministic result or failure."""
        self.definitions, self.result = definitions, result
        self.calls = []

    async def discover_tools(self):
        """Return server definitions without granting their use."""
        return self.definitions

    async def call(self, name, arguments):
        """Capture dispatch without performing external operations."""
        self.calls.append((name, arguments))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def bind(client, run):
    """Use the production discovery and binding path."""
    collector = EvidenceCollector()
    harness = RunHarness(run)
    tools = await create_mcp_tools(client, start=START, end=END,
                                   collector=collector, harness=harness)
    return {tool.name: tool for tool in tools}, collector, harness


def test_policy_intersects_agent_grants_and_task_without_expansion():
    grants = ("catalog.lookup", "read_evidence", "catalog.lookup")
    assert resolve_tool_policy(grants) == ("catalog.lookup", "read_evidence")
    assert resolve_tool_policy(grants, agent_tools=(
        "catalog.lookup", "ungranted_tool")) == ("catalog.lookup",)
    assert resolve_tool_policy(grants, agent_tools=()) == ()
    assert resolve_tool_policy(grants, purpose=RunPurpose.EXPLAIN,
                               agent_tools=("catalog.lookup",)) == ()
    run = state(grants, purpose=RunPurpose.EXPLAIN)
    RunHarness(run)
    assert run.allowed_tools == ("read_evidence",)


@pytest.mark.parametrize("name", ["", "has space", "x" * 129, "bad/name"])
def test_invalid_tool_names_are_rejected_in_configuration_and_state(name):
    from tracemind.config import Settings

    with pytest.raises(ValidationError):
        state((name,))
    with pytest.raises(ValidationError):
        Settings(_env_file=None, agent_allowed_tools=(name,))


def test_old_and_new_grants_survive_storage_without_migration(tmp_path):
    repository = RunRepository(tmp_path / "runs.db")
    for grants in (TOOL_NAMES, ("catalog.lookup", "read_evidence")):
        run = state(grants, status=RunStatus.QUEUED)
        repository.create(run, request_hash=str(run.id))
        restored = repository.get(run.id).state
        assert restored.allowed_tools == grants
        assert RunState.model_validate_json(
            restored.model_dump_json()).allowed_tools == grants


@pytest.mark.parametrize("structured", [None, {
    "id": "reference-id", "evidence": [{"id": "not-an-observation"}],
}])
def test_generic_text_json_are_references_and_new_names_need_no_registry(
    structured,
):
    result = CallToolResult(
        content=[TextContent(type="text", text="Runbook instructions")],
        structuredContent=structured,
    )
    client = Client([definition("catalog.lookup"), definition("hidden")],
                    result)

    async def exercise():
        tools, collector, harness = await bind(
            client, state(("catalog.lookup",)))
        assert set(tools) == {"catalog.lookup"}
        message = await tools["catalog.lookup"].ainvoke({
            "name": "catalog.lookup", "args": {}, "id": "call-1",
            "type": "tool_call",
        })
        assert "context_index" not in message.artifact
        assert ("reference-id" if structured else "Runbook instructions") in (
            message.content)
        assert collector.evidence == []
        assert harness.state.evidence_ids == []
        assert harness.state.progress_count == 1
        completed = next(e for e in harness.state.events
                         if e.kind == "tool.completed")
        assert completed.data["evidence_ids"] == []
        assert harness.state.backend_attempts == 1
        harness.state.allowed_tools = ()
        with pytest.raises(RunStopped) as rejected:
            await tools["catalog.lookup"].ainvoke({})
        assert rejected.value.reason is StopReason.TOOL_DENIED
        assert len(client.calls) == 1

    asyncio.run(exercise())


def test_http_client_passes_generic_results_to_the_binding(monkeypatch):
    result = CallToolResult(content=[TextContent(type="text", text="hello")])
    client = HttpMcpClient("http://unused")

    class Session:
        async def call_tool(self, name, arguments):
            return result

    async def connect():
        return Session()

    monkeypatch.setattr(client, "_connect", connect)
    assert asyncio.run(client.call("catalog.lookup", {})) is result


@pytest.mark.parametrize("legacy_schema", [False, True])
def test_renamed_telemetry_tool_reaches_validated_diagnosis(legacy_schema):
    observation = Evidence(source="log", observed_at=END,
                           summary="Payment Charge failed")
    declared = definition("inspect.service", meta=(None if legacy_schema else {
        RESULT_CONTRACT_KEY: TELEMETRY_CONTRACT,
    }))
    if legacy_schema:
        declared.outputSchema = TelemetryReply.model_json_schema()
    client = Client([declared], CallToolResult(content=[], structuredContent={
        "evidence": [observation.model_dump(mode="json")],
    }))
    model = ScriptedModel(respond=lambda messages, calls: (
        tool_call("inspect.service", {"service": "payment"}) if calls == 1
        else diagnosis(messages)))
    service = DiagnosisService(LangChainAgentRunner(model),
                               tool_client=client,
                               allowed_tools=("inspect.service",))
    run = asyncio.run(service.diagnose_with_evidence(
        INCIDENT, start=START, end=END))
    assert run.state.status is RunStatus.SUCCEEDED
    assert run.diagnosis.evidence_ids == [observation.id]
    assert client.calls == [("inspect.service", {"service": "payment"})]


@pytest.mark.parametrize("empty", [False, True])
def test_queries_explain_no_new_evidence_without_counting_feedback(empty):
    observation = Evidence(source="log", observed_at=END,
                           summary="Payment Charge failed")
    declared = definition("inspect.service", meta={
        RESULT_CONTRACT_KEY: TELEMETRY_CONTRACT,
    })
    client = Client([declared], CallToolResult(content=[], structuredContent={
        "evidence": ([] if empty else [observation.model_dump(mode="json")]),
    }))

    async def exercise():
        tools, collector, harness = await bind(
            client, state(("inspect.service",)))
        results = []
        for index in range(2):
            message = await tools["inspect.service"].ainvoke({
                "name": "inspect.service", "args": {},
                "id": f"query-{index}", "type": "tool_call",
            })
            results.append(json.loads(message.content))
        feedback = results[-1][-1]
        assert feedback["kind"] == "query_feedback"
        assert feedback["new_evidence"] == 0
        assert harness.state.progress_count == (0 if empty else 1)
        assert len(collector.evidence) == (0 if empty else 1)
        assert len(client.calls) == 2

    asyncio.run(exercise())


def test_generic_reference_id_cannot_become_a_diagnosis_citation():
    observation = Evidence(source="log", observed_at=END, summary="Reference")
    client = Client([definition("catalog.lookup")], CallToolResult(
        content=[], structuredContent=observation.model_dump(mode="json")))
    model = ScriptedModel(respond=lambda messages, calls: (
        tool_call("catalog.lookup", {"query": "runbook"}) if calls == 1
        else diagnosis(messages)))
    service = DiagnosisService(LangChainAgentRunner(model), tool_client=client,
                               allowed_tools=("catalog.lookup",))
    run = asyncio.run(service.diagnose_with_evidence(
        INCIDENT, start=START, end=END))
    assert run.state.stop_reason is StopReason.INVALID_DIAGNOSIS
    assert run.evidence == []
    assert not any(e.kind == "diagnosis.accepted" for e in run.state.events)


@pytest.mark.parametrize("definitions,grants", [
    ([definition("read_evidence")], ("catalog.lookup",)),
    ([definition("DiagnosisDraft")], ("DiagnosisDraft",)),
    ([definition("catalog.lookup"), definition("catalog.lookup")],
     ("catalog.lookup",)),
    ([definition("other")], ("catalog.lookup",)),
    ([definition("catalog.lookup", meta={RESULT_CONTRACT_KEY: "unknown"})],
     ("catalog.lookup",)),
])
def test_bad_discovery_contracts_fail_before_dispatch(definitions, grants):
    client = Client(definitions, CallToolResult(content=[]))
    with pytest.raises(RunStopped) as rejected:
        asyncio.run(bind(client, state(grants)))
    assert rejected.value.reason is StopReason.TOOL_ERROR
    assert client.calls == []


def test_generic_transient_failure_is_not_implicitly_retried():
    client = Client([definition("catalog.lookup")], RemoteTelemetryError(
        "transient", retryable=True))

    async def exercise():
        tools, _, harness = await bind(client, state(("catalog.lookup",)))
        with pytest.raises(ToolUnavailable):
            await tools["catalog.lookup"].ainvoke({})
        assert harness.state.tool_calls == {"catalog.lookup": 1}
        assert len(client.calls) == 1

    asyncio.run(exercise())


@pytest.mark.parametrize("result,category", [
    (CallToolResult(content=[TextContent(type="text", text="private error")],
                    isError=True), "invalid_request"),
    (CallToolResult(content=[ImageContent(
        type="image", data="AA==", mimeType="image/png")]),
     "unsupported_content"),
])
def test_generic_errors_and_unsupported_content_are_not_observations(
    result, category,
):
    client = Client([definition("catalog.lookup")], result)

    async def exercise():
        tools, collector, harness = await bind(
            client, state(("catalog.lookup",)))
        expected = ToolUnavailable if result.isError else RunStopped
        with pytest.raises(expected):
            await tools["catalog.lookup"].ainvoke({})
        assert collector.evidence == []
        failed = next(e for e in harness.state.events
                      if e.kind == "tool.failed")
        assert failed.data["category"] == category
        assert "private error" not in str(harness.state.events)

    asyncio.run(exercise())
