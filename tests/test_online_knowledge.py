"""Verify online RAG controls using fixed retrieval and scripted models."""

import asyncio
import json
from uuid import uuid4

import httpx
from langchain_core.messages import AIMessage, ToolMessage
import pytest
from test_harness import END, INCIDENT, START, ScriptedModel, tool_call

from tracemind.agent.errors import DiagnosisValidationError
from tracemind.agent.runtime.harness import RunHarness, RunStopped
from tracemind.agent.runtime.knowledge import (
    create_knowledge_tools, knowledge_context,
)
from tracemind.agent.runtime.recovery import ToolUnavailable
from tracemind.agent.runtime.runner import LangChainAgentRunner
from tracemind.agent.runtime.tools import EvidenceCollector, create_mcp_tools
from tracemind.agent.service import DiagnosisService
from tracemind.agent.storage.run_repository import RunRepository
from tracemind.agent.workflow.validation import validate_draft
from tracemind.models import Diagnosis, DiagnosisStatus
from tracemind.models.knowledge import KnowledgePolicy, KnowledgeReference
from tracemind.models.result import DiagnosisDraft
from tracemind.models.run import (
    RunLimits, RunPurpose, RunState, RunStatus, StopReason,
)
from tracemind.rag.models import KnowledgeChunk, KnowledgeParent, RetrievalHit


POLICY = KnowledgePolicy(snapshot="fixture-v1")
GRANTS = ("search_knowledge", "read_knowledge")


def hit(text="An RPC timeout may indicate a slow dependency."):
    """Return canonical fixture content with source revision and hash."""
    return RetrievalHit(chunk=KnowledgeChunk(
        chunk_id="rpc-timeout", doc_id="rpc-runbook", title="RPC runbook",
        section="Timeouts", text=text, document_path="runbooks/rpc.md",
        document_sha256="f" * 64,
        source_refs=["https://example.test/docs/rpc"],
        source_revision="fixture-rev-1",
    ), branches=["mix", "bm25"], fusion_score=.03, rerank_score=.8)


def parent_hit():
    """Keep a complete section whose tail is absent from the matched child."""
    result = hit()
    result.chunk.parent_id = "timeouts-section"
    text = result.chunk.text + "\n" + "Extra context. " * 250 + "Section tail."
    result.parent = KnowledgeParent(
        **result.chunk.model_dump(exclude={
            "chunk_id", "parent_id", "source_start", "source_end",
            "source_text", "identifiers", "text"}),
        parent_id=result.chunk.parent_id, text=text,
        source_start=0, source_end=len(text))
    return result


def test_parent_section_is_paged_losslessly_from_retained_state():
    retrieval = Retrieval()
    retrieval.hits = [parent_hit()]
    state = run_state()
    _, tools = tools_for(state, retrieval)

    async def exercise():
        references = await tools["search_knowledge"].ainvoke({"query": "RPC"})
        pages, offset = [], 0
        while offset is not None:
            page = await tools["read_knowledge"].ainvoke({
                "reference_id": references[0]["knowledge_ref"],
                "offset": offset})
            pages.append(page[0]["content"])
            offset = page[0]["next_offset"]
        assert "".join(pages) == retrieval.hits[0].parent.text

    asyncio.run(exercise())
    assert state.model_calls == 0
    restored = RunState.model_validate_json(state.model_dump_json())
    assert restored.knowledge[0].hit.parent == retrieval.hits[0].parent


def test_auxiliary_reservation_preserves_final_synthesis_after_retry():
    attempts = []

    async def completion(*args, **kwargs):
        attempts.append(True)
        raise httpx.ConnectError("temporary provider failure")

    model = ScriptedModel(respond=lambda messages, calls: (
        tool_call("search_knowledge", {"query": "RPC timeout"}) if calls == 1
        else answer(messages)))
    result = asyncio.run(service(
        model, Retrieval(keywords=True), knowledge_completion=completion,
        limits=RunLimits(model_calls=3, retry_backoff_seconds=0),
    ).diagnose_with_evidence(INCIDENT, start=START, end=END))
    assert len(attempts) == 1
    assert result.state.model_calls == 3
    assert model.calls == 2
    assert result.state.stop_reason is StopReason.INSUFFICIENT_EVIDENCE
    assert any(e.kind == "model.skipped" and e.data.get("reason") ==
               "model_budget_reserved" for e in result.state.events)


def test_knowledge_compression_keeps_reader_contract_and_latest_parent_page():
    from test_context import request_with_results
    from tracemind.agent.workflow.context import estimate_input, select_context

    state = run_state()
    _, tools = tools_for(state, Retrieval())
    request, _ = request_with_results()
    reference = KnowledgeReference(snapshot=POLICY.snapshot, hit=parent_hit())
    index = [{"knowledge_ref": reference.id, "title": "Timeouts"},
             {"error": "partial_search"}]
    messages = [request.messages[0],
                tool_call("search_knowledge", {"query": "RPC"}),
                ToolMessage(content=json.dumps([{"text": "x" * 5000}]),
                            tool_call_id="search", name="search_knowledge",
                            artifact={"context_index": index,
                                      "context_reference_key": "knowledge_ref",
                                      "context_reader": "read_knowledge"}),
                tool_call("read_knowledge", {"reference_id": reference.id}),
                ToolMessage(content="parent section page", tool_call_id="read",
                            name="read_knowledge", artifact={
                                "context_index": index, "detail_page": True,
                                "context_reader": "read_knowledge"})]
    request = request.override(messages=messages, tools=list(tools.values()))
    selection = select_context(request, estimate_input(request) - 4000)
    assert selection.fits
    assert json.loads(selection.request.messages[2].content) == index
    assert selection.request.messages[-1].content == "parent section page"


def run_state(**kwargs):
    """Build an isolated opted-in run without environment configuration."""
    return RunState(incident=INCIDENT, investigation_start=START,
                    investigation_end=END, allowed_tools=GRANTS,
                    limits=kwargs.pop("limits", RunLimits()),
                    knowledge_policy=kwargs.pop("knowledge_policy", POLICY),
                    **kwargs)


class NoRemoteTools:
    """Fail if a local-only investigation attempts MCP discovery."""

    transport = "replay"

    async def discover_tools(self):
        raise AssertionError("local knowledge does not require MCP discovery")

    async def call(self, name, arguments):
        raise AssertionError("no remote tool was admitted")


class Retrieval:
    """Record snapshot queries and optionally exercise internal completion."""

    def __init__(self, *, keywords=False, error=None):
        self.queries = []
        self.keywords, self.error = keywords, error
        self.hits = [hit()]

    async def __call__(self, queries, *, question, snapshot, completion):
        assert snapshot == POLICY.snapshot
        self.queries.extend(queries)
        if self.error:
            raise self.error
        if self.keywords:
            await completion([{"role": "user", "content": question}],
                             max_tokens=2048, keyword_extraction=True)
        return self.hits


def tools_for(state, search=None, completion=None):
    """Bind the real local tools to a run harness."""
    harness = RunHarness(state)
    tools = create_knowledge_tools(harness, search=search,
                                   completion=completion,
                                   model_name="fixture-auxiliary")
    return harness, {t.name: t for t in tools}


def answer(messages, *, complete=False, references=None):
    """Construct a proposal from references the model actually received."""
    refs = references
    if refs is None:
        refs = []
        for message in messages:
            if isinstance(message, ToolMessage) and message.status != "error":
                payload = json.loads(message.content)
                refs.extend(item["knowledge_ref"] for item in payload
                            if "knowledge_ref" in item)
    return tool_call("DiagnosisDraft", {
        "status": "complete" if complete else "inconclusive",
        "root_cause": "Documentation suggests a check; no fault is confirmed.",
        "confidence": .1, "knowledge_refs": list(dict.fromkeys(refs)),
    })


def service(model, search=None, **kwargs):
    """Build the production graph without a live telemetry or RAG runtime."""
    return DiagnosisService(LangChainAgentRunner(model),
                            tool_client=NoRemoteTools(), allowed_tools=GRANTS,
                            knowledge_policy=POLICY, knowledge_search=search,
                            **kwargs)


def test_optional_retrieval_never_forces_tool_calls():
    retrieval = Retrieval()
    model = ScriptedModel(respond=lambda messages, calls: answer(messages))
    result = asyncio.run(service(model, retrieval).diagnose_with_evidence(
        INCIDENT, start=START, end=END))
    assert retrieval.queries == []
    assert result.state.model_calls == 1
    assert result.tool_calls == {}
    assert result.state.knowledge == []


def test_factory_wires_enabled_search_without_expanding_grants(monkeypatch):
    from tracemind.agent.factory import build_service
    from tracemind.config import Settings
    import tracemind.rag.retriever as retriever
    import tracemind.agent.factory as factory

    retrieval = Retrieval()
    monkeypatch.setattr(retriever, "get_snapshot_retriever",
                        lambda config: retrieval)
    monkeypatch.setattr(factory, "create_rag_completion",
                        lambda settings, rag: None)
    configured = Settings(_env_file=None, agent_knowledge_enabled=True,
                          agent_knowledge_snapshot=POLICY.snapshot,
                          agent_allowed_tools=GRANTS,
                          agent_trace_directory=None)
    runner = LangChainAgentRunner(ScriptedModel(
        respond=lambda messages, calls: (
            tool_call("search_knowledge", {"query": "RPC timeout"})
            if calls == 1 else answer(messages))))
    selected = build_service(configured, runner, NoRemoteTools())
    result = asyncio.run(selected.diagnose_with_evidence(
        INCIDENT, start=START, end=END))
    assert retrieval.queries == ["RPC timeout"]
    assert result.state.allowed_tools == GRANTS
    assert result.state.knowledge_policy == configured.knowledge_policy


def test_english_search_preserves_provenance_dedup_and_copies():
    retrieval = Retrieval()
    state = run_state()
    harness, tools = tools_for(state, retrieval)

    async def exercise():
        first = await tools["search_knowledge"].ainvoke({
            "query": "RPC timeout",
        })
        second = await tools["search_knowledge"].ainvoke({"query": "slow RPC"})
        assert first == second
        assert first[0]["source_revision"] == "fixture-rev-1"
        assert first[0]["document_sha256"] == "f" * 64
        page = await tools["read_knowledge"].ainvoke({
            "reference_id": first[0]["knowledge_ref"],
        })
        assert "slow dependency" in page[0]["content"]

    asyncio.run(exercise())
    assert len(state.knowledge) == 1
    retrieval.hits[0].chunk.text = "changed externally"
    assert "changed externally" not in state.knowledge[0].hit.chunk.text
    assert state.evidence_ids == []
    assert state.model_calls == 0
    assert all(e.data["evidence_ids"] == [] for e in state.events
               if e.kind == "tool.completed")
    assert harness.state.backend_attempts == 2


def test_planning_and_keyword_calls_share_usage_budget_and_trace():
    retrieval = Retrieval(keywords=True)
    auxiliary_calls = []

    async def completion(messages, *, max_tokens, keyword_extraction):
        auxiliary_calls.append(keyword_extraction)
        assert max_tokens == POLICY.model_output_tokens
        content = ("{}" if keyword_extraction else
                   '{"queries":["RPC timeout"]}')
        return AIMessage(content=content, usage_metadata={
            "input_tokens": 7, "output_tokens": 3, "total_tokens": 10,
        })

    model = ScriptedModel(respond=lambda messages, calls: (
        tool_call("search_knowledge", {"query": "解释 RPC 超时", "rewrite": True})
        if calls == 1 else answer(messages)))
    result = asyncio.run(service(
        model, retrieval, knowledge_completion=completion,
        knowledge_model_name="fixture-auxiliary",
        limits=RunLimits(model_calls=4),
    ).diagnose_with_evidence(INCIDENT, start=START, end=END))
    assert auxiliary_calls == [False, True]
    assert result.state.model_calls == 4
    assert result.state.usage.reported_model_calls == 4
    assert result.state.usage.reported_total_tokens == 50
    assert result.state.usage.unreported_model_calls == 0
    assert result.diagnosis.knowledge_refs == [result.state.knowledge[0].id]
    assert result.evidence == []
    tool_span = next(e.span_id for e in result.state.events
                     if e.kind == "tool.started")
    auxiliary = [e for e in result.state.events if e.kind == "model.started"
                 and e.data["purpose"].startswith("knowledge.")]
    assert {e.data["purpose"] for e in auxiliary} == {
        "knowledge.query_planning", "knowledge.keyword_extraction",
    }
    assert all(e.parent_span_id == tool_span for e in auxiliary)


def test_shared_model_limit_blocks_auxiliary_before_provider_execution():
    retrieval = Retrieval()
    model = ScriptedModel(respond=lambda messages, calls: tool_call(
        "search_knowledge", {"query": "translate this", "rewrite": True}))

    async def completion(*args, **kwargs):
        raise AssertionError("shared model budget exhausted")

    result = asyncio.run(service(
        model, retrieval, knowledge_completion=completion,
        limits=RunLimits(model_calls=1),
    ).diagnose_with_evidence(INCIDENT, start=START, end=END))
    assert result.state.stop_reason is StopReason.MODEL_BUDGET
    assert result.state.model_calls == 1
    assert retrieval.queries == []


@pytest.mark.parametrize("failure", [RuntimeError("private body"),
                                     OSError("private connection")])
def test_retrieval_failure_degrades_and_allows_the_agent_to_finish(failure):
    model = ScriptedModel(respond=lambda messages, calls: (
        tool_call("search_knowledge", {"query": "RPC timeout"}) if calls == 1
        else answer(messages)))
    result = asyncio.run(service(model, Retrieval(error=failure))
                         .diagnose_with_evidence(
                             INCIDENT, start=START, end=END))
    assert result.state.stop_reason is StopReason.INSUFFICIENT_EVIDENCE
    assert result.state.model_calls == 2
    assert result.state.knowledge == []
    assert any(e.kind == "tool.unavailable" for e in result.state.events)
    assert "private" not in str(result.state.events)


def test_invalid_query_plan_returns_a_tool_error_without_retrieval():
    retrieval = Retrieval()

    async def completion(*args, **kwargs):
        return AIMessage(content='{"queries":[" "]}')

    state = run_state()
    _, tools = tools_for(state, retrieval, completion)
    with pytest.raises(ToolUnavailable):
        asyncio.run(tools["search_knowledge"].ainvoke({
            "query": "translate", "rewrite": True,
        }))
    assert state.model_calls == 1
    assert state.usage.unreported_model_calls == 1
    assert retrieval.queries == []


def test_auxiliary_context_limit_blocks_hidden_model_calls():
    async def retrieval(queries, *, question, snapshot, completion):
        await completion([{"role": "user", "content": "x" * 5000}],
                         max_tokens=512, keyword_extraction=True)
        raise AssertionError("oversize internal prompt must be rejected")

    async def completion(*args, **kwargs):
        raise AssertionError("provider must not see the oversize prompt")

    state = run_state(limits=RunLimits(context_tokens=2000))
    _, tools = tools_for(state, retrieval, completion)
    with pytest.raises(ToolUnavailable) as caught:
        asyncio.run(tools["search_knowledge"].ainvoke({
            "query": "RPC timeout",
        }))
    assert caught.value.failure.category == "knowledge_context_limit"
    assert state.model_calls == 0


def test_tool_timeout_does_not_retry_or_leave_retained_hits():
    calls = []

    async def retrieval(query, **kwargs):
        calls.append(query)
        await asyncio.sleep(1)
        return [hit()]

    state = run_state(limits=RunLimits(tool_timeout_seconds=.01))
    _, tools = tools_for(state, retrieval)
    with pytest.raises(ToolUnavailable):
        asyncio.run(tools["search_knowledge"].ainvoke({
            "query": "RPC timeout",
        }))
    assert calls == [["RPC timeout"]]
    assert state.knowledge == []
    assert state.tool_calls == {"search_knowledge": 1}


def test_cancellation_closes_nested_model_and_tool_spans():
    async def exercise():
        entered = asyncio.Event()

        async def completion(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        state = run_state()
        _, tools = tools_for(state, Retrieval(keywords=True), completion)
        task = asyncio.create_task(tools["search_knowledge"].ainvoke({
            "query": "RPC timeout",
        }))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state.model_calls == 1
        assert state.usage.unreported_model_calls == 1
        assert state.knowledge == []
        assert sum(e.kind == "model.cancelled" for e in state.events) == 1
        assert sum(e.kind == "tool.cancelled" for e in state.events) == 1

    asyncio.run(exercise())


def test_capacity_excludes_unretained_references_and_bounds_context():
    state = run_state(knowledge_policy=POLICY.model_copy(update={
        "retention_bytes": 1,
    }))
    _, tools = tools_for(state, Retrieval())
    result = asyncio.run(tools["search_knowledge"].ainvoke({"query": "RPC"}))
    assert result == [{"omitted_count": 1}]
    assert state.knowledge == []
    state.knowledge = [KnowledgeReference(snapshot=POLICY.snapshot,
                                          hit=hit("x" * 5000))]
    state.knowledge_policy.context_bytes = 100
    context = knowledge_context(state)
    assert len(context.encode()) <= 100
    assert json.loads(context)["omitted"] == 1


@pytest.mark.parametrize("refs,complete,code", [
    (["unknown"], False, "unknown_knowledge"),
    (["known", "known"], False, "duplicate_knowledge"),
    (["known"], True, "missing_evidence"),
])
def test_knowledge_citations_are_validated_separately(refs, complete, code):
    reference = KnowledgeReference(snapshot=POLICY.snapshot, hit=hit())
    selected = [reference.id if ref == "known" else ref for ref in refs]
    draft = DiagnosisDraft(status="complete" if complete else "inconclusive",
                           confidence=.1, root_cause="Documented mechanism",
                           knowledge_refs=selected)
    with pytest.raises(DiagnosisValidationError) as caught:
        validate_draft(draft, [], start=START, end=END, knowledge=[reference])
    assert caught.value.code == code
    if code != "missing_evidence":
        assert caught.value.details()["knowledge_refs"] == selected


def test_documentation_alone_cannot_complete_a_real_graph_diagnosis():
    model = ScriptedModel(respond=lambda messages, calls: (
        tool_call("search_knowledge", {"query": "RPC timeout"}) if calls == 1
        else answer(messages, complete=True)))
    result = asyncio.run(service(model, Retrieval()).diagnose_with_evidence(
        INCIDENT, start=START, end=END))
    assert result.state.stop_reason is StopReason.INVALID_DIAGNOSIS
    assert len(result.state.knowledge) == 1
    assert result.evidence == []


def test_grants_are_rechecked_after_local_tool_binding():
    state = run_state()
    _, tools = tools_for(state, Retrieval())
    state.allowed_tools = ()
    with pytest.raises(RunStopped) as caught:
        asyncio.run(tools["search_knowledge"].ainvoke({"query": "RPC"}))
    assert caught.value.reason is StopReason.TOOL_DENIED
    assert state.tool_calls == {}


def test_explanation_restores_references_without_new_retrieval(tmp_path):
    repository = RunRepository(tmp_path / "runs.db")
    parent = run_state(status=RunStatus.QUEUED, investigation_id=uuid4())
    repository.create(parent, request_hash="initial", new_investigation=True)
    claim = repository.claim_next()
    original = claim.run.state
    reference = KnowledgeReference(snapshot=POLICY.snapshot, hit=parent_hit())
    original.knowledge = [reference]
    original.diagnosis = Diagnosis(
        incident_id=INCIDENT.id, status=DiagnosisStatus.INCONCLUSIVE,
        root_cause="Documented mechanism remains unconfirmed", confidence=.1,
        knowledge_refs=[reference.id],
    )
    original.status = RunStatus.INCONCLUSIVE
    original.stop_reason = StopReason.INSUFFICIENT_EVIDENCE
    repository.finish(original.id, claim.owner, original, [])
    follow = repository.follow_up(
        parent.investigation_id, message="Explain the documentation",
        purpose=RunPurpose.EXPLAIN, limits=RunLimits(), allowed_tools=GRANTS,
        request_hash="explain",
    )
    assert follow.state.allowed_tools == ("read_knowledge",)
    assert follow.state.knowledge[0].id == reference.id

    model = ScriptedModel(respond=lambda messages, calls: (
        tool_call("read_knowledge", {"reference_id": reference.id})
        if calls == 1 else answer(messages)))
    retrieval = Retrieval(error=AssertionError("explanation cannot retrieve"))
    selected = service(model, retrieval)
    claim = repository.claim_next()
    result = asyncio.run(selected.run_persisted(
        claim.run.state, [],
        persist=lambda state, evidence: repository.save_progress(
            state.id, claim.owner, state, evidence),
        is_cancelled=lambda: False, resume=False, graph_resume=False,
    ))
    repository.finish(result.state.id, claim.owner,
                      result.state, result.evidence)
    restored = repository.get(result.state.id)
    assert restored.result.diagnosis.knowledge_refs == [reference.id]
    assert restored.state.knowledge[0].hit.chunk == reference.hit.chunk
    assert restored.state.knowledge[0].hit.parent == reference.hit.parent
    assert retrieval.queries == []
    assert restored.state.backend_attempts == 0


def test_old_snapshots_have_empty_knowledge_and_no_implicit_grants():
    from tracemind.config import Settings

    old = run_state().model_dump(mode="json", exclude={
        "knowledge", "knowledge_policy",
    })
    restored = RunState.model_validate(old)
    assert restored.knowledge == []
    assert restored.knowledge_policy is None
    settings = Settings(_env_file=None)
    assert not set(GRANTS).intersection(settings.agent_allowed_tools)


def test_injected_local_tools_are_filtered_before_mcp_discovery():
    async def exercise():
        state = run_state()
        harness, local = tools_for(state, Retrieval())
        state.allowed_tools = ("read_knowledge",)
        tools = await create_mcp_tools(
            NoRemoteTools(), start=START, end=END,
            collector=EvidenceCollector(), harness=harness,
            local_tools=list(local.values()),
        )
        assert [t.name for t in tools] == ["read_knowledge"]

    asyncio.run(exercise())


def test_parallel_internal_calls_keep_distinct_tool_span_parents():
    async def exercise():
        both_entered = asyncio.Event()
        calls = []

        async def completion(messages, **kwargs):
            calls.append(messages)
            if len(calls) == 2:
                both_entered.set()
            await both_entered.wait()
            return AIMessage(content="{}", usage_metadata={
                "input_tokens": 7, "output_tokens": 3, "total_tokens": 10,
            })

        state = run_state()
        _, tools = tools_for(state, Retrieval(keywords=True), completion)
        await asyncio.gather(*[tools["search_knowledge"].ainvoke({"query": q})
                               for q in ("RPC timeout", "slow dependency")])
        assert state.model_calls == 2
        assert state.usage.reported_total_tokens == 20
        assert len(state.knowledge) == 1
        tool_spans = {e.span_id for e in state.events
                      if e.kind == "tool.started"}
        parents = {e.parent_span_id for e in state.events
                   if e.kind == "model.started"}
        assert len(tool_spans) == 2
        assert parents == tool_spans

    asyncio.run(exercise())


def test_optional_model_failure_keeps_attempts_and_allows_diagnosis():
    async def completion(*args, **kwargs):
        raise httpx.ConnectError("private provider body")

    model = ScriptedModel(respond=lambda messages, calls: (
        tool_call("search_knowledge", {"query": "RPC timeout"}) if calls == 1
        else answer(messages)))
    result = asyncio.run(service(
        model, Retrieval(keywords=True), knowledge_completion=completion,
        limits=RunLimits(retry_attempts=0),
    ).diagnose_with_evidence(INCIDENT, start=START, end=END))
    assert result.state.stop_reason is StopReason.INSUFFICIENT_EVIDENCE
    assert result.state.model_calls == 3
    assert result.state.usage.unreported_model_calls == 1
    assert result.state.usage.reported_model_calls == 2
    assert "private provider body" not in str(result.state.events)


def test_changed_canonical_chunk_is_rejected_in_the_same_snapshot():
    retrieval = Retrieval()
    state = run_state()
    _, tools = tools_for(state, retrieval)

    async def exercise():
        await tools["search_knowledge"].ainvoke({"query": "RPC timeout"})
        retrieval.hits = [hit("Different content for the same canonical ID")]
        with pytest.raises(ToolUnavailable):
            await tools["search_knowledge"].ainvoke({"query": "slow RPC"})

    asyncio.run(exercise())
    assert state.knowledge[0].hit.chunk.text == hit().chunk.text


def test_knowledge_and_observations_can_be_cited_together_in_the_graph():
    from test_harness import Telemetry
    from tool_fixtures import FixtureToolClient

    def respond(messages, calls):
        if calls == 1:
            metrics = tool_call("query_metrics", {"expression": "up"})
            knowledge = tool_call("search_knowledge", {"query": "RPC timeout"})
            return metrics.model_copy(update={
                "tool_calls": [*metrics.tool_calls, *knowledge.tool_calls],
            })
        observations, refs = [], []
        for message in messages:
            if isinstance(message, ToolMessage) and message.status != "error":
                for item in json.loads(message.content):
                    if "id" in item:
                        observations.append(item["id"])
                    if "knowledge_ref" in item:
                        refs.append(item["knowledge_ref"])
        return tool_call("DiagnosisDraft", {
            "status": "complete", "confidence": .8,
            "root_cause": (
                "Observed payment errors; documentation informs checks"),
            "evidence_ids": observations, "knowledge_refs": refs,
        })

    telemetry = Telemetry()
    selected = DiagnosisService(
        LangChainAgentRunner(ScriptedModel(respond=respond)),
        tool_client=FixtureToolClient(telemetry, telemetry, telemetry),
        allowed_tools=("query_metrics", *GRANTS), knowledge_policy=POLICY,
        knowledge_search=Retrieval(),
    )
    result = asyncio.run(selected.diagnose_with_evidence(
        INCIDENT, start=START, end=END))
    assert result.state.status is RunStatus.SUCCEEDED
    assert result.diagnosis.evidence_ids == [result.evidence[0].id]
    assert result.diagnosis.knowledge_refs == [result.state.knowledge[0].id]
