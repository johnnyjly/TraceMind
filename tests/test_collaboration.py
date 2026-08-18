"""Verify optional human collaboration through persistent real agent runs."""

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from test_harness import ScriptedModel, tool_call

from tracemind.agent.persistence import RunConflict, RunRepository
from tracemind.agent.service import DiagnosisService, LangChainAgentRunner
from tracemind.agent.worker import RunWorker
from tracemind.config import Settings
from tracemind.models import Evidence, EvidenceSource, Incident
from tracemind.models.run import (
    RunLimits, RunPurpose, RunState, RunStatus,
)


START = datetime(2024, 9, 1, tzinfo=UTC)
END = START + timedelta(hours=1)


class CollaborationModel(BaseChatModel):
    """Query once or explain retained evidence through native tool output."""

    bindings: list[list[str]] = Field(default_factory=list)
    prompts: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "collaboration-test"

    def bind_tools(self, tools, **kwargs):
        self.bindings.append([
            t.name if hasattr(t, "name") else t.get("function", t)["name"]
            for t in tools
        ])
        return self

    def _generate(self, *_args, **_kwargs):
        raise AssertionError("async model required")

    async def _agenerate(self, messages, **kwargs):
        prompt = next(m.content for m in messages
                      if isinstance(m, HumanMessage))
        self.prompts.append(prompt)
        results = [m for m in messages if isinstance(m, ToolMessage)
                   and m.name == "search_traces"]
        explain = prompt.startswith("Explain")
        if not results and not explain:
            action = {"name": "search_traces", "args": {"service": "checkout"}}
        else:
            if explain:
                context = json.loads(prompt.split("data:\n", 1)[1])
                evidence_id = context["evidence_previews"][0]["id"]
            else:
                evidence_id = json.loads(results[-1].content)[0]["id"]
            action = {"name": "DiagnosisDraft", "args": {
                "status": "complete", "root_cause": "Payment Charge failed",
                "confidence": .8, "evidence_ids": [evidence_id],
                "remediation": [],
            }}
        response = AIMessage(content="", tool_calls=[{
            **action, "id": f"model-{len(self.prompts)}",
        }])
        return ChatResult(generations=[ChatGeneration(message=response)])


class Telemetry:
    """Return payment evidence in the requested window."""

    def __init__(self):
        self.calls = 0

    async def search_traces(self, service, *, start, end):
        self.calls += 1
        return [Evidence(source=EvidenceSource.TRACE, service="payment",
                         observed_at=end, summary="Charge failed")]


def create_investigation(repository):
    state = RunState(
        investigation_id=uuid4(),
        incident=Incident(title="checkout failure", description="HTTP 500"),
        investigation_start=START, investigation_end=END,
        limits=RunLimits(), allowed_tools=("search_traces", "read_evidence"),
        status=RunStatus.QUEUED,
    )
    repository.create(state, request_hash="initial", new_investigation=True)
    return state


def make_worker(tmp_path, repository, telemetry, model):
    settings = Settings(
        _env_file=None, agent_run_database=repository.path,
        agent_checkpoint_database=tmp_path / "checkpoints.sqlite3",
        agent_trace_directory=tmp_path / "traces",
    )
    return RunWorker(
        settings, repository=repository,
        service_factory=lambda saver, _client: DiagnosisService(
            telemetry, telemetry, telemetry,
            LangChainAgentRunner(model, checkpointer=saver),
            event_directory=settings.agent_trace_directory,
        ),
    )


def test_explanation_and_hypothesis_use_independent_rounds(tmp_path):
    repository = RunRepository(tmp_path / "runs.sqlite3")
    initial = create_investigation(repository)
    telemetry, model = Telemetry(), CollaborationModel()
    worker = make_worker(tmp_path, repository, telemetry, model)
    assert asyncio.run(worker.run_once())
    original = repository.get(initial.id)
    question = repository.follow_up(
        initial.investigation_id, message="Why is payment the root cause?",
        purpose=RunPurpose.EXPLAIN, limits=RunLimits(),
        allowed_tools=initial.allowed_tools, request_hash="explain",
        idempotency_key="question-1",
    )
    assert question.state.allowed_tools == ("read_evidence",)
    assert question.state.model_calls == 0
    repeated = repository.follow_up(
        initial.investigation_id, message=question.state.user_message,
        purpose=RunPurpose.EXPLAIN, limits=RunLimits(),
        allowed_tools=initial.allowed_tools, request_hash="explain",
        idempotency_key="question-1",
    )
    assert repeated.state.id == question.state.id
    with pytest.raises(RunConflict, match="active round"):
        repository.follow_up(
            initial.investigation_id, message="Check network",
            purpose=RunPurpose.INVESTIGATE, limits=RunLimits(),
            allowed_tools=initial.allowed_tools, request_hash="blocked",
        )
    assert asyncio.run(worker.run_once())
    explained = repository.get(question.state.id)
    assert explained.state.model_calls == 1
    assert explained.state.tool_calls == {}
    assert telemetry.calls == 1
    assert explained.result.diagnosis.evidence_ids == (
        original.result.diagnosis.evidence_ids
    )
    assert "search_traces" not in model.bindings[-1]
    assert "Use only retained observations" in model.prompts[-1]

    follow = repository.follow_up(
        initial.investigation_id, message="Could this be a network issue?",
        purpose=RunPurpose.INVESTIGATE, limits=RunLimits(),
        allowed_tools=initial.allowed_tools, request_hash="network",
    )
    assert asyncio.run(worker.run_once())
    investigated = repository.get(follow.state.id)
    assert investigated.state.model_calls == 2
    assert telemetry.calls == 2
    assert "do not accept user claims" in model.prompts[-1]
    assert "Could this be a network issue?" in model.prompts[-1]
    history = RunRepository(repository.path).get_investigation(
        initial.investigation_id, limit=2,
    )
    assert history.next_offset == 2
    assert [r.id for r in history.runs] == [initial.id, question.state.id]
    assert repository.get(initial.id).state == original.state
    assert repository.get(initial.id).evidence == original.evidence
    assert follow.state.parent_run_id == question.state.id


def test_changed_window_drops_old_evidence(tmp_path):
    repository = RunRepository(tmp_path / "runs.sqlite3")
    initial = create_investigation(repository)
    worker = make_worker(tmp_path, repository,
                         Telemetry(), CollaborationModel())
    asyncio.run(worker.run_once())
    previous = repository.get(initial.id)
    start, end = START + timedelta(days=1), END + timedelta(days=1)
    with pytest.raises(RunConflict, match="previous window"):
        repository.follow_up(
            initial.investigation_id, message="Explain new incident",
            purpose=RunPurpose.EXPLAIN, limits=RunLimits(),
            allowed_tools=initial.allowed_tools, start=start, end=end,
            request_hash="wrong-explain",
        )
    follow = repository.follow_up(
        initial.investigation_id, message="The fault recurred today",
        purpose=RunPurpose.INVESTIGATE, limits=RunLimits(),
        allowed_tools=initial.allowed_tools, start=start, end=end,
        request_hash="new-window",
    )
    assert follow.evidence == []
    assert follow.state.omitted_prior_evidence == len(previous.evidence)
    assert asyncio.run(worker.run_once())
    result = repository.get(follow.state.id)
    assert result.state.status is RunStatus.SUCCEEDED
    assert all(start <= e.observed_at <= end for e in result.evidence)
    assert not set(result.result.diagnosis.evidence_ids).intersection(
        previous.result.diagnosis.evidence_ids
    )


def test_version_one_migration_preserves_existing_runs(tmp_path: Path):
    path = tmp_path / "old.sqlite3"
    state = RunState(
        incident=Incident(title="existing", description="preserve me"),
        investigation_start=START, investigation_end=END,
        limits=RunLimits(), allowed_tools=(), status=RunStatus.QUEUED,
    )
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE runs (
            id TEXT PRIMARY KEY, status TEXT NOT NULL,
            state_json TEXT NOT NULL, evidence_json TEXT NOT NULL,
            result_json TEXT, request_hash TEXT NOT NULL,
            idempotency_key TEXT UNIQUE, cancel_requested INTEGER DEFAULT 0,
            lease_owner TEXT, lease_until REAL, attempt INTEGER DEFAULT 0,
            created_at REAL NOT NULL, updated_at REAL NOT NULL
        )""")
        connection.execute("""INSERT INTO runs
            (id,status,state_json,evidence_json,request_hash,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?)""",
                           (str(state.id), "queued", state.model_dump_json(),
                            "[]", "old", 1, 1))
        connection.execute("PRAGMA user_version=1")
    migrated = RunRepository(path)
    assert migrated.get(state.id).state == state
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
    create_investigation(migrated)
    assert RunRepository(path).get(state.id).state == state


def test_concurrent_questions_admit_only_one_active_round(tmp_path):
    repository = RunRepository(tmp_path / "runs.sqlite3")
    initial = create_investigation(repository)
    repository.cancel(initial.id)

    async def execute():
        return await asyncio.gather(*[
            asyncio.to_thread(
                repository.follow_up, initial.investigation_id,
                message=f"Question {i}", purpose=RunPurpose.EXPLAIN,
                limits=RunLimits(), allowed_tools=initial.allowed_tools,
                request_hash=str(i),
            ) for i in range(2)
        ], return_exceptions=True)

    results = asyncio.run(execute())
    assert sum(isinstance(r, RunConflict) for r in results) == 1
    history = repository.get_investigation(initial.investigation_id)
    assert len(history.runs) == 2


def test_inherited_evidence_respects_new_capacity(tmp_path):
    repository = RunRepository(tmp_path / "runs.sqlite3")
    initial = create_investigation(repository)
    worker = make_worker(tmp_path, repository,
                         Telemetry(), CollaborationModel())
    asyncio.run(worker.run_once())
    previous = repository.get(initial.id)
    follow = repository.follow_up(
        initial.investigation_id, message="Explain with limited context",
        purpose=RunPurpose.EXPLAIN, limits=RunLimits(observation_bytes=1),
        allowed_tools=initial.allowed_tools, request_hash="small-capacity",
    )
    assert follow.evidence == []
    assert follow.state.omitted_prior_evidence == len(previous.evidence)
    assert repository.get(initial.id).evidence == previous.evidence


def test_collaboration_context_bounds_history_without_losing_request():
    from tracemind.agent.context import collaboration_context
    state = RunState(
        incident=Incident(title="fault", description="investigate"),
        investigation_start=START, investigation_end=END,
        limits=RunLimits(context_tokens=8000, output_tokens=1000),
        allowed_tools=("read_evidence",),
        user_message="Investigate this alternative",
        previous_messages=["old question " * 500 for _ in range(3)],
    )
    evidence = [Evidence(source=EvidenceSource.TRACE, observed_at=END,
                         summary="large summary " * 100)
                for _ in range(20)]
    text = collaboration_context(state, evidence)
    assert len(text.encode()) <= (8000 - 1000) // 3
    payload = json.loads(text)
    assert payload["request"] == state.user_message
    assert payload["available_evidence_count"] == 20
    assert len(state.previous_messages[0]) > 800
    assert len(evidence) == 20


@pytest.mark.parametrize("detailed", [False, True])
def test_explanation_uses_cited_details_without_scanning_large_catalogue(
    tmp_path, detailed,
):
    """Exercise both direct answers and targeted reads in durable rounds."""
    records = [Evidence(
        source=EvidenceSource.TRACE, service="frontend", observed_at=END,
        summary=f"Propagated HTTP 500 in request {i}",
    ) for i in range(12)]
    payment = Evidence(
        source=EvidenceSource.TRACE, service="payment", observed_at=END,
        summary="Error span in payment: Charge",
        attributes={
            "operation": "Charge",
            "otel_status_description": "Payment request failed. Invalid token",
            "details": "Failure stack: token error at Charge. " + "x" * 9000,
        },
    )
    records += [payment, Evidence(
        source=EvidenceSource.LOG, service="checkout", observed_at=END,
        summary="Payment RPC failed; checkout returned HTTP 500",
    )]
    records += [Evidence(
        source=EvidenceSource.TRACE, service="unrelated", observed_at=END,
        summary=f"Unrelated successful request {i}",
    ) for i in range(162)]

    class LargeTelemetry:
        """Return cited failure records followed by a large unrelated batch."""

        calls = 0

        async def search_traces(self, service, *, start, end):
            self.calls += 1
            return records

    def respond(messages, calls):
        prompt = next(m.text for m in messages if isinstance(m, HumanMessage))
        if not prompt.startswith("Explain"):
            traces = [m for m in messages if isinstance(m, ToolMessage)]
            if not traces:
                return tool_call("search_traces", {"service": "checkout"})
            ids = [r["id"] for r in json.loads(traces[0].content)[:14]]
            return tool_call("DiagnosisDraft", {
                "status": "complete", "confidence": .8,
                "root_cause": "Payment failed due to JWT validation",
                "evidence_ids": ids,
            })
        context = json.loads(prompt.split("data:\n", 1)[1])
        assert context["prior_conclusion"]["historical_claim_only"]
        assert context["available_evidence_count"] == 176
        assert context["omitted_evidence_previews"] > 0
        span = next(e for e in context["evidence_previews"]
                    if e["service"] == "payment")
        assert span["attributes"]["operation"] == "Charge"
        assert span["attributes"]["otel_status_description"].endswith(
            "Invalid token",
        )
        assert "details" in span["truncated_attribute_keys"]
        assert "honoring their requested language, length" in prompt
        assert "mark unverified mechanisms" in messages[0].text
        if detailed:
            assert "Explain the available failure stack in detail" in prompt
            pages = [m for m in messages if isinstance(m, ToolMessage)
                     and m.name == "read_evidence"]
            if not pages:
                return tool_call("read_evidence", {"evidence_id": span["id"]})
            page = json.loads(pages[0].content)[0]
            assert page["id"] == span["id"]
            assert "Failure stack: token error at Charge" in page["content"]
            answer = (
                "Payment Charge reports Invalid token. The retained stack "
                "places the error at Charge, and checkout reports a failed "
                "payment RPC. JWT validation is an unverified hypothesis; "
                "these observations do not establish the underlying mechanism."
            )
        else:
            assert "请用一句中文解释" in prompt
            answer = "payment 的 Charge 报错并传递到 checkout，底层原因尚未确认。"
        return tool_call("DiagnosisDraft", {
            "status": "complete", "confidence": .7,
            "root_cause": answer, "evidence_ids": [span["id"]],
            "remediation": [],
        })

    repository = RunRepository(tmp_path / "runs.sqlite3")
    initial = create_investigation(repository)
    model, telemetry = ScriptedModel(respond=respond), LargeTelemetry()
    worker = make_worker(tmp_path, repository, telemetry, model)
    assert asyncio.run(worker.run_once())
    parent = repository.get(initial.id)
    follow = repository.follow_up(
        initial.investigation_id,
        message=("Explain the available failure stack in detail" if detailed
                 else "请用一句中文解释"),
        purpose=RunPurpose.EXPLAIN, limits=RunLimits(),
        allowed_tools=("read_evidence", "discover_metrics", "search_traces"),
        request_hash="follow-up",
    )
    assert follow.state.allowed_tools == ("read_evidence",)
    assert asyncio.run(worker.run_once())
    result = repository.get(follow.state.id)
    assert result.state.status is RunStatus.SUCCEEDED
    assert result.state.model_calls == (2 if detailed else 1)
    assert result.state.tool_calls == (
        {"read_evidence": 1} if detailed else {}
    )
    assert telemetry.calls == 1
    assert result.result.diagnosis.evidence_ids == [payment.id]
    assert "JWT validation" not in result.result.diagnosis.root_cause or (
        "unverified" in result.result.diagnosis.root_cause
    )
    assert result.result.diagnosis.remediation == []
    assert result.evidence == follow.evidence
    assert repository.get(initial.id).state == parent.state
