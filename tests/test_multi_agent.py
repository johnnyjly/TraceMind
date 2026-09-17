"""Real graphs with controlled models: concurrency, provenance and recovery."""

import asyncio
import json
from uuid import uuid4

import pytest
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from test_harness import (
    END,
    INCIDENT,
    START,
    ScriptedModel,
    Telemetry,
    tool_call,
)
from test_verifier import EVIDENCE, ReviewModel, review
from proposal_fixtures import proposal_payload
from tool_fixtures import FixtureToolClient

from tracemind.agent.errors import DiagnosisError, DiagnosisValidationError
from tracemind.agent.runtime.audit import trace_issues
from tracemind.agent.runtime.collaboration import (
    merge_completed,
    coordinator_context,
    validate_report,
)
from tracemind.agent.runtime.harness import RunHarness
from tracemind.agent.runtime.runner import LangChainAgentRunner
from tracemind.agent.service import DiagnosisService
from tracemind.agent.verification import DiagnosisVerifier
from tracemind.config import Settings
from tracemind.models import Evidence
from tracemind.models.collaboration import (
    CollaborationState,
    InvestigationReport,
    InvestigationTask,
    TaskRecord,
)
from tracemind.models.run import (
    RunLimits,
    RunState,
    RunStatus,
    StopReason,
    TOOL_NAMES,
)

TASK = {
    "role": "telemetry",
    "question": "Which operation failed?",
    "hypothesis_ids": [],
    "distinguishing_observations": ["Linked payment errors or successes"],
}
REPORT = {
    "findings": [
        {
            "text": "Payment Charge failed",
            "kind": "observation",
            "evidence_refs": ["E1"],
        }
    ],
    "missing_observations": ["Underlying mechanism"],
}
DRAFT = {
    "status": "complete",
    "root_cause": "Payment Charge failed",
    "confidence": 0.8,
    "evidence_ids": ["E1"],
}


class FixedTelemetry(Telemetry):
    async def query(self, expression, *, at=None):
        await super().query(expression, at=at)
        return [EVIDENCE]


def coordinator_packet(messages):
    if "Return CoordinatorDecision" not in messages[0].text:
        return None
    return json.loads(messages[1].text.split("Coordinator context:\n")[1])


def needs_investigation(messages):
    packet = coordinator_packet(messages)
    return (
        packet is not None
        and not packet["tasks"]
        and packet["can_investigate"]
    )


def dispatch_tasks(payload):
    return tool_call(
        "CoordinatorDecision",
        {
            "action": "investigate",
            "reason": "Test the missing operation",
            "hypotheses": [],
            "proposal": None,
            **payload,
        },
    )


def submit(proposal):
    return tool_call(
        "CoordinatorDecision",
        {
            "action": "submit",
            "tasks": [],
            "hypotheses": [],
            "reason": "Evidence answers the task",
            "proposal": proposal_payload(proposal),
        },
    )


def response(messages, calls):
    if needs_investigation(messages):
        return dispatch_tasks({"tasks": [TASK]})
    if "You are an investigator" in messages[0].text:
        if not any(isinstance(m, ToolMessage) for m in messages):
            return tool_call()
        return tool_call("InvestigationReport", REPORT)
    packet = coordinator_packet(messages)
    assert packet is not None
    if packet["originals"]:
        assert packet["originals"][0]["id"] == "E1"
        return submit(DRAFT)
    return submit(
        {
            "status": "inconclusive",
            "root_cause": "No operation observed",
            "confidence": 0,
            "evidence_ids": [],
        }
    )


def service_for(
    monkeypatch,
    *,
    respond=response,
    limits=None,
    telemetry=None,
    saver=None,
    reviewer=None,
):
    settings = Settings(
        _env_file=None, agent_mode="multi", agent_trace_directory=None
    )
    model = ScriptedModel(respond=respond)
    reviewer = reviewer or ReviewModel(body=review())
    monkeypatch.setattr(
        "tracemind.agent.verification.create_review_model",
        lambda settings: reviewer,
    )
    telemetry = telemetry or FixedTelemetry()
    service = DiagnosisService(
        LangChainAgentRunner(model, checkpointer=saver),
        agent_mode="multi",
        limits=limits or RunLimits(model_calls=12),
        tool_client=FixtureToolClient(telemetry, telemetry, telemetry),
        verification_policy=settings.verification_policy,
        verifier=DiagnosisVerifier(settings),
    )
    return service, model, reviewer, telemetry


def run(service):
    return asyncio.run(
        service.diagnose_with_evidence(INCIDENT, start=START, end=END)
    )


def test_graph_uses_shared_budget_originals_and_review(monkeypatch):
    service, model, reviewer, telemetry = service_for(monkeypatch)
    result = run(service)
    assert result.state.status == RunStatus.SUCCEEDED
    assert result.state.model_calls == model.calls + reviewer.calls == 5
    assert result.state.collaboration.phase == "completed"
    assert len(telemetry.calls) == 1
    assert trace_issues(result.state) == []


def test_parallel_tasks_overlap_and_coalesce_identical_query(monkeypatch):
    entered = set()
    ready = None

    async def respond(messages, calls):
        nonlocal ready
        if needs_investigation(messages):
            return dispatch_tasks(
                {
                    "tasks": [
                        TASK,
                        {
                            **TASK,
                            "question": "Which operations still succeed?",
                        },
                    ]
                },
            )
        if "You are an investigator" in messages[0].text and not any(
            isinstance(m, ToolMessage) for m in messages
        ):
            ready = ready or asyncio.Event()
            entered.add(messages[1].text)
            if len(entered) == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), 1)
        return response(messages, calls)

    service, model, reviewer, telemetry = service_for(
        monkeypatch, respond=respond, limits=RunLimits(model_calls=20)
    )
    result = run(service)
    state = result.state
    assert len(entered) == len(state.collaboration.tasks) == 2
    assert len(telemetry.calls) == state.backend_attempts == 1
    assert state.model_calls == model.calls + reviewer.calls == 7
    assert any(e.kind == "tool.reused" for e in state.events)
    assert all(
        t.status == "completed" for t in state.collaboration.tasks.values()
    )
    task_ids = set(state.collaboration.tasks)
    assert {
        e.data["task_id"]
        for e in state.events
        if e.kind == "model.started" and e.data.get("role") == "telemetry"
    } == task_ids
    assert trace_issues(state) == []


def test_supported_and_counter_reference_overlap_between_claims_is_legal():
    state = RunState(
        incident=INCIDENT,
        allowed_tools=TOOL_NAMES,
        limits=RunLimits(),
        investigation_start=START,
        investigation_end=END,
        reference_map={"E1": str(EVIDENCE.id)},
    )
    harness = RunHarness(state)
    harness.validate_draft = lambda _: None
    report = InvestigationReport.model_validate(
        {
            **REPORT,
            "findings": [
                REPORT["findings"][0],
                {
                    "text": "Not every step failed",
                    "kind": "hypothesis",
                    "counterevidence_refs": ["E1"],
                },
            ],
        }
    )
    validate_report(report, state, harness)


@pytest.mark.parametrize("refs", [["missing"], ["K1"], ["E1", "E1"]])
def test_report_rejects_unknown_wrong_type_and_duplicate_refs(refs):
    state = RunState(
        incident=INCIDENT,
        allowed_tools=TOOL_NAMES,
        limits=RunLimits(),
        investigation_start=START,
        investigation_end=END,
        reference_map={"E1": str(EVIDENCE.id), "K1": "knowledge"},
    )
    harness = RunHarness(state)
    harness.validate_draft = lambda _: None
    report = InvestigationReport.model_validate(
        {
            **REPORT,
            "findings": [{**REPORT["findings"][0], "evidence_refs": refs}],
        }
    )
    with pytest.raises(DiagnosisValidationError):
        validate_report(report, state, harness)


@pytest.mark.parametrize("calls", [1, 3, 4, 5, 6, 8, 12])
def test_model_cap_is_never_exceeded(monkeypatch, calls):
    service, model, reviewer, _ = service_for(
        monkeypatch, limits=RunLimits(model_calls=calls)
    )
    try:
        state = run(service).state
    except DiagnosisError as exc:
        state = exc.state
    assert state.model_calls <= calls
    assert model.calls + reviewer.calls == state.model_calls
    assert trace_issues(state) == []


def test_semantic_rejection_routes_only_gaps_to_new_tasks(monkeypatch):
    plans = 0
    reviewer = ReviewModel(body=review("insufficient"))

    def respond(messages, calls):
        nonlocal plans
        if needs_investigation(messages):
            plans += 1
            if plans > 1:
                assert "A linked failing operation" in messages[1].text
                reviewer.body = review()
            return dispatch_tasks(
                {
                    "tasks": [
                        {
                            **TASK,
                            "question": f"Check operation round {plans}",
                        }
                    ]
                },
            )
        return response(messages, calls)

    service, _, _, _ = service_for(
        monkeypatch,
        respond=respond,
        reviewer=reviewer,
        limits=RunLimits(model_calls=24),
    )
    result = run(service)
    assert plans == 2
    assert result.state.collaboration.round == 1
    assert len(result.state.collaboration.reviews) == 1
    assert reviewer.calls == 2
    assert result.state.verification.review.accepted
    assert result.state.status == RunStatus.SUCCEEDED


def test_feedback_rounds_are_bounded_and_partial_results_preserved(
    monkeypatch,
):
    count = 0

    def respond(messages, calls):
        nonlocal count
        if needs_investigation(messages):
            count += 1
            return dispatch_tasks(
                {"tasks": [{**TASK, "question": f"Check cohort {count}"}]},
            )
        return response(messages, calls)

    service, _, reviewer, _ = service_for(
        monkeypatch,
        respond=respond,
        reviewer=ReviewModel(body=review("insufficient")),
        limits=RunLimits(model_calls=40),
    )
    result = run(service)
    assert count == reviewer.calls == 3
    assert result.state.collaboration.round == 2
    assert result.state.stop_reason == StopReason.VERIFICATION_REJECTED


def test_reducer_is_idempotent_and_detects_conflicting_writes():
    assert merge_completed({"a": "x"}, {"a": "x", "b": "y"}) == {
        "a": "x",
        "b": "y",
    }
    with pytest.raises(ValueError, match="conflicting"):
        merge_completed({"a": "x"}, {"a": "y"})


def test_handoff_includes_more_than_twenty_cited_originals():
    evidence = [
        Evidence(
            source="trace",
            service="payment",
            observed_at=END,
            summary=f"Operation {i}",
        )
        for i in range(30)
    ]
    state = RunState(
        incident=INCIDENT,
        allowed_tools=TOOL_NAMES,
        limits=RunLimits(),
        investigation_start=START,
        investigation_end=END,
        evidence_ids=[e.id for e in evidence],
        reference_map={f"E{i + 1}": str(e.id) for i, e in enumerate(evidence)},
        collaboration=CollaborationState(),
    )
    findings = [
        {
            "text": f"Scope {i}",
            "kind": "observation",
            "evidence_refs": [
                f"E{j + 1}" for j in range(i * 10, (i + 1) * 10)
            ],
        }
        for i in range(3)
    ]
    state.collaboration.tasks["task"] = TaskRecord(
        tool_limit=10,
        id="task",
        round=0,
        task=InvestigationTask(**TASK),
        status="completed",
        report=InvestigationReport(findings=findings, missing_observations=[]),
    )
    packet = coordinator_context(state, evidence)
    assert len(packet["originals"]) == 30
    assert packet["omitted_original_refs"] == []


def test_completed_run_resumes_without_paid_calls(monkeypatch, tmp_path):
    async def exercise():
        async with AsyncSqliteSaver.from_conn_string(
            str(tmp_path / "graph.db")
        ) as saver:
            service, model, reviewer, _ = service_for(monkeypatch, saver=saver)
            original = await service.diagnose_with_evidence(
                INCIDENT, start=START, end=END
            )
            before = model.calls + reviewer.calls
            state = original.state.model_copy(deep=True)
            state.status = RunStatus.RUNNING
            state.stop_reason = None
            state.events = [
                e for e in state.events if e.kind != "run.finished"
            ]
            restored = await service.run_persisted(
                state,
                original.evidence,
                persist=lambda *args: None,
                is_cancelled=lambda: False,
                resume=True,
                graph_resume=True,
            )
            assert restored.diagnosis == original.diagnosis
            assert model.calls + reviewer.calls == before

    asyncio.run(exercise())


def test_shared_cache_survives_ledger_roundtrip_and_is_run_local():
    async def exercise():
        state = RunState(
            incident=INCIDENT,
            allowed_tools=TOOL_NAMES,
            limits=RunLimits(),
            investigation_start=START,
            investigation_end=END,
            collaboration=CollaborationState(),
        )
        calls = 0

        async def fetch():
            nonlocal calls
            calls += 1
            return [{"value": 3}]

        first = RunHarness(state)
        assert await first.shared_tool_call(
            "query_metrics", {"q": "x"}, fetch
        ) == [{"value": 3}]
        second = RunHarness(
            RunState.model_validate_json(state.model_dump_json())
        )
        cached = await second.shared_tool_call(
            "query_metrics", {"q": "x"}, fetch
        )
        cached[0]["value"] = 99
        assert calls == 1
        assert await second.shared_tool_call(
            "query_metrics", {"q": "x"}, fetch
        ) == [{"value": 3}]
        isolated = RunHarness(
            RunState(
                incident=INCIDENT,
                allowed_tools=TOOL_NAMES,
                limits=RunLimits(),
                investigation_start=START,
                investigation_end=END,
                collaboration=CollaborationState(),
            )
        )
        await isolated.shared_tool_call("query_metrics", {"q": "x"}, fetch)
        assert calls == 2

    asyncio.run(exercise())


def test_cancelled_owner_releases_shared_query_waiters():
    async def exercise():
        state = RunState(
            incident=INCIDENT,
            allowed_tools=TOOL_NAMES,
            limits=RunLimits(),
            investigation_start=START,
            investigation_end=END,
            collaboration=CollaborationState(),
        )
        harness = RunHarness(state)
        started = asyncio.Event()

        async def fetch():
            started.set()
            await asyncio.Future()

        owner = asyncio.create_task(
            harness.shared_tool_call("query_metrics", {}, fetch)
        )
        await started.wait()
        waiter = asyncio.create_task(
            harness.shared_tool_call("query_metrics", {}, fetch)
        )
        await asyncio.sleep(0)
        owner.cancel()
        results = await asyncio.wait_for(
            asyncio.gather(owner, waiter, return_exceptions=True), 1
        )
        assert all(isinstance(x, asyncio.CancelledError) for x in results)
        assert harness._inflight_tools == {}

    asyncio.run(exercise())


def test_knowledge_agent_uses_pinned_snapshot_and_separate_citations(
    monkeypatch,
):
    from test_online_knowledge import POLICY, Retrieval
    from tracemind.models.knowledge import KnowledgePolicy

    retrieval = Retrieval()

    def respond(messages, calls):
        if needs_investigation(messages):
            return dispatch_tasks(
                {
                    "tasks": [
                        TASK,
                        {
                            **TASK,
                            "role": "knowledge",
                            "question": "Explain RPC semantics",
                        },
                    ]
                },
            )
        if (
            "You are an investigator" in messages[0].text
            and '"role":"knowledge"' in messages[1].text
        ):
            if not any(isinstance(m, ToolMessage) for m in messages):
                return tool_call("search_knowledge", {"query": "RPC timeout"})
            return tool_call(
                "InvestigationReport",
                {
                    "findings": [
                        {
                            "text": "RPC timeout may indicate a slow "
                            "dependency",
                            "kind": "knowledge",
                            "knowledge_refs": ["K1"],
                        }
                    ],
                    "missing_observations": [],
                },
            )
        if coordinator_packet(
            messages
        ) is not None and not needs_investigation(messages):
            assert "K1" in messages[1].text
        return response(messages, calls)

    service, _, _, _ = service_for(
        monkeypatch, respond=respond, limits=RunLimits(model_calls=24)
    )
    service._knowledge_policy = KnowledgePolicy.model_validate(
        POLICY.model_dump()
    )
    service._knowledge_search = retrieval
    service._allowed_tools = (
        *TOOL_NAMES,
        "search_knowledge",
        "read_knowledge",
    )
    result = run(service)
    assert len(result.state.knowledge) == 1
    assert result.state.knowledge[0].snapshot == POLICY.snapshot
    assert result.state.reference_map["K1"] == result.state.knowledge[0].id
    roles = {r.task.role for r in result.state.collaboration.tasks.values()}
    assert roles == {"knowledge", "telemetry"}
    assert result.state.verification.review.accepted


def test_crash_between_report_publication_and_graph_commit_reuses_output(
    monkeypatch, tmp_path
):
    async def exercise():
        async with AsyncSqliteSaver.from_conn_string(
            str(tmp_path / "graph.db")
        ) as saver:
            service, model, reviewer, telemetry = service_for(
                monkeypatch, saver=saver
            )
            state = RunState(
                incident=INCIDENT,
                investigation_start=START,
                investigation_end=END,
                limits=RunLimits(model_calls=12),
                allowed_tools=TOOL_NAMES,
                agent_mode="multi",
                verification_policy=Settings(
                    _env_file=None
                ).verification_policy,
            )
            saved = []

            def persist(current, evidence):
                workflow = current.collaboration
                if (
                    workflow
                    and any(
                        t.status == "completed"
                        for t in workflow.tasks.values()
                    )
                    and not saved
                ):
                    saved.append(
                        (current.model_copy(deep=True), list(evidence))
                    )
                    raise RuntimeError("simulated process interruption")

            with pytest.raises(DiagnosisError):
                await service.run_persisted(
                    state,
                    [],
                    persist=persist,
                    is_cancelled=lambda: False,
                    resume=False,
                    graph_resume=False,
                )
            restored, evidence = saved[0]
            before = model.calls
            result = await service.run_persisted(
                restored,
                evidence,
                persist=lambda *args: None,
                is_cancelled=lambda: False,
                resume=True,
                graph_resume=True,
            )
            assert result.state.status == RunStatus.SUCCEEDED
            # Synthesis only, no repeated task.
            assert model.calls == before + 1
            assert reviewer.calls == 1
            assert len(telemetry.calls) == 1

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "failure", ["model_timeout", "tool_timeout", "denied"]
)
def test_parallel_investigation_failure_is_bounded(monkeypatch, failure):
    async def respond(messages, calls):
        if "You are an investigator" in messages[0].text:
            if failure == "model_timeout":
                await asyncio.Future()
            if failure == "denied":
                return tool_call("delete_service", {})
        return response(messages, calls)

    class SlowTelemetry(FixedTelemetry):
        async def query(self, expression, *, at=None):
            await asyncio.Future()

    service, model, reviewer, _ = service_for(
        monkeypatch,
        respond=respond,
        telemetry=SlowTelemetry() if failure == "tool_timeout" else None,
        limits=RunLimits(
            model_calls=12,
            model_timeout_seconds=0.1,
            tool_timeout_seconds=0.01,
            retry_attempts=0,
        ),
    )
    try:
        state = run(service).state
    except DiagnosisError as exc:
        state = exc.state
    assert state.model_calls == model.calls + reviewer.calls <= 12
    assert state.status != RunStatus.SUCCEEDED
    assert trace_issues(state) == []
    if failure == "denied":
        assert state.stop_reason == StopReason.TOOL_DENIED
        assert state.backend_attempts == 0


def test_real_process_exit_resumes_completed_task(monkeypatch, tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    child = tmp_path / "child.py"
    child.write_text("""
import asyncio, json, os
from pathlib import Path
from pytest import MonkeyPatch
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from test_multi_agent import service_for, INCIDENT, START, END
from tracemind.config import Settings
from tracemind.models.run import RunState, RunLimits, TOOL_NAMES
ROOT = Path(os.environ["PROCESS_TEST_DIR"])
async def main():
    database = str(ROOT / "graph.db")
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        service, _, _, _ = service_for(MonkeyPatch(), saver=saver)
        state = RunState(incident=INCIDENT, investigation_start=START,
            investigation_end=END, limits=RunLimits(model_calls=12),
            allowed_tools=TOOL_NAMES, agent_mode="multi",
            verification_policy=Settings(_env_file=None).verification_policy)
        def persist(current, evidence):
            workflow = current.collaboration
            if workflow and any(t.status == "completed"
                                for t in workflow.tasks.values()):
                (ROOT/"saved.json").write_text(json.dumps({
                    "state":current.model_dump(mode="json"),
                    "evidence":[e.model_dump(mode="json") for e in evidence]}))
                os._exit(17)
        await service.run_persisted(state, [], persist=persist,
            is_cancelled=lambda: False, resume=False, graph_resume=False)
asyncio.run(main())
""")
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PROCESS_TEST_DIR": str(tmp_path),
        "PYTHONPATH": os.pathsep.join(
            [
                str(root / "src"),
                str(root / "tests"),
                str(root / "services/otel-telemetry-mcp/src"),
            ]
        ),
    }
    result = subprocess.run(
        [sys.executable, str(child)], env=env, capture_output=True, timeout=20
    )
    assert result.returncode == 17, result.stderr.decode()
    saved = json.loads((tmp_path / "saved.json").read_text())
    state = RunState.model_validate(saved["state"])
    evidence = [Evidence.model_validate(e) for e in saved["evidence"]]
    assert state.model_calls == 3

    async def restore():
        async with AsyncSqliteSaver.from_conn_string(
            str(tmp_path / "graph.db")
        ) as saver:
            service, model, reviewer, telemetry = service_for(
                monkeypatch, saver=saver
            )
            recovered = await service.run_persisted(
                state,
                evidence,
                persist=lambda *args: None,
                is_cancelled=lambda: False,
                resume=True,
                graph_resume=True,
            )
            assert recovered.state.status == RunStatus.SUCCEEDED
            assert recovered.state.model_calls == 5
            assert model.calls == reviewer.calls == 1
            assert len(telemetry.calls) == 0

    asyncio.run(restore())


def test_invalid_specialist_at_reserve_preserves_synthesis(monkeypatch):
    synthesis_attempts = 0

    def incomplete(messages, calls):
        nonlocal synthesis_attempts
        if "You are an investigator" in messages[0].text:
            if not any(isinstance(m, ToolMessage) for m in messages):
                return tool_call()
            return tool_call(
                "InvestigationReport",
                {
                    "findings": [
                        {
                            "text": "Unsupported observation",
                            "kind": "observation",
                        }
                    ],
                    "missing_observations": [],
                },
            )
        if coordinator_packet(
            messages
        ) is not None and not needs_investigation(messages):
            synthesis_attempts += 1
            if synthesis_attempts == 1:
                return submit(
                    {
                        **DRAFT,
                        "evidence_ids": ["E999"],
                    }
                )
        return response(messages, calls)

    service, _, _, _ = service_for(
        monkeypatch, respond=incomplete, limits=RunLimits(model_calls=8)
    )
    result = asyncio.run(
        service.diagnose_with_evidence(INCIDENT, start=START, end=END)
    )
    assert result.state.model_calls <= 8
    assert result.state.verification.status == "completed"
    assert synthesis_attempts == 2
    assert any(
        e.kind == "collaboration.task_incomplete" for e in result.state.events
    )
    task = next(iter(result.state.collaboration.tasks.values()))
    assert task.report.findings == []
    assert result.evidence


@pytest.mark.parametrize(
    "transition", ["dispatch", "join", "submit", "feedback", "complete"]
)
def test_durable_transitions_do_not_repeat_completed_decisions(
    monkeypatch, tmp_path, transition
):
    reviewer = ReviewModel(body=review("insufficient"))

    def responding(messages, calls):
        packet = coordinator_packet(messages)
        if packet and packet["review_feedback"]:
            reviewer.body = review()
            assert packet["review_feedback"][0]["item_id"]
            return submit(DRAFT)
        return response(messages, calls)

    async def exercise():
        async with AsyncSqliteSaver.from_conn_string(
            str(tmp_path / "transitions.db")
        ) as saver:
            service, model, _, telemetry = service_for(
                monkeypatch,
                respond=responding,
                reviewer=reviewer,
                saver=saver,
                limits=RunLimits(model_calls=20),
            )
            state = RunState(
                incident=INCIDENT,
                investigation_start=START,
                investigation_end=END,
                limits=RunLimits(model_calls=20),
                allowed_tools=TOOL_NAMES,
                agent_mode="multi",
                verification_policy=Settings(
                    _env_file=None
                ).verification_policy,
            )
            saved = []

            def persist(current, evidence):
                w = current.collaboration
                if not w or saved:
                    return
                matches = {
                    "dispatch": w.phase == "investigate",
                    "join": w.phase == "coordinate"
                    and w.decision == 1
                    and bool(w.tasks),
                    "submit": w.phase == "review",
                    "feedback": w.phase == "coordinate" and w.round == 1,
                    "complete": w.phase == "completed",
                }
                if matches[transition]:
                    saved.append(
                        (current.model_copy(deep=True), list(evidence))
                    )
                    raise RuntimeError("interrupted after durable transition")

            with pytest.raises(DiagnosisError):
                await service.run_persisted(
                    state,
                    [],
                    persist=persist,
                    is_cancelled=lambda: False,
                    resume=False,
                    graph_resume=False,
                )
            result = await service.run_persisted(
                *saved[0],
                persist=lambda *args: None,
                is_cancelled=lambda: False,
                resume=True,
                graph_resume=True,
            )
            assert result.state.status == RunStatus.SUCCEEDED
            assert (
                model.calls + reviewer.calls == result.state.model_calls == 7
            )
            assert len(telemetry.calls) == 1
            assert len(result.state.collaboration.reviews) == 1
            assert result.state.collaboration.round == 1

    asyncio.run(exercise())


def test_two_parallel_specialists_preserve_success_after_sibling_timeout(
    monkeypatch,
):
    entered = set()
    barrier = None

    async def responding(messages, calls):
        nonlocal barrier
        if needs_investigation(messages):
            return dispatch_tasks(
                {"tasks": [TASK, {**TASK, "question": "fail"}]}
            )
        if "You are an investigator" in messages[0].text:
            question = json.loads(
                messages[1]
                .text.split("Assigned task:\n")[1]
                .split("\nAssigned hypotheses:")[0]
            )["question"]
            if not any(isinstance(m, ToolMessage) for m in messages):
                barrier = barrier or asyncio.Event()
                entered.add(question)
                if len(entered) == 2:
                    barrier.set()
                await barrier.wait()
            if question == "fail":
                await asyncio.Future()
        return response(messages, calls)

    service, model, reviewer, telemetry = service_for(
        monkeypatch,
        respond=responding,
        limits=RunLimits(
            model_calls=20, model_timeout_seconds=0.1, retry_attempts=0
        ),
    )
    result = run(service)
    assert len(entered) == 2
    assert sorted(
        t.status for t in result.state.collaboration.tasks.values()
    ) == ["completed", "failed"]
    assert result.state.status == RunStatus.SUCCEEDED
    assert len(result.evidence) == len(telemetry.calls) == 1
    assert result.state.model_calls == model.calls + reviewer.calls


def test_old_workflow_state_is_not_silently_resumed():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CollaborationState.model_validate({"version": "graph-v3"})


@pytest.mark.parametrize(
    "count,cap,admitted",
    [(1, 7, False), (1, 8, True), (2, 9, False), (2, 10, True)],
)
def test_exact_investigation_admission_boundaries(
    monkeypatch, count, cap, admitted
):
    def proposing(messages, calls):
        packet = coordinator_packet(messages)
        if packet is not None and not packet["tasks"]:
            return dispatch_tasks(
                {
                    "tasks": [
                        {**TASK, "question": f"Operation check {i}"}
                        for i in range(count)
                    ]
                }
            )
        return response(messages, calls)

    service, model, reviewer, telemetry = service_for(
        monkeypatch, respond=proposing, limits=RunLimits(model_calls=cap)
    )
    try:
        state = run(service).state
    except DiagnosisError as exc:
        state = exc.state
    assert state.model_calls <= cap
    if admitted:
        assert len(state.collaboration.tasks) == count
        assert all(
            t.status == "completed" for t in state.collaboration.tasks.values()
        )
        assert len(telemetry.calls) == 1
        assert state.model_calls == 3 + 2 * count
    else:
        assert not state.collaboration.tasks
        assert not telemetry.calls
        error = ("task_allocation" if count == 2 and cap == 9
                 else "investigation_closed")
        assert any(
            error in json.dumps(e.model_dump(mode="json"))
            for e in state.events
        )


def test_completed_child_before_ledger_application_is_not_recharged(
    monkeypatch, tmp_path
):
    from tracemind.agent.runtime.collaboration import MultiAgentRunner

    original = MultiAgentRunner._invoke
    saved = []

    async def interrupt(
        self,
        key,
        role,
        system,
        prompt,
        schema,
        tools,
        harness,
        recursion_limit,
        **kwargs,
    ):
        result = await original(
            self,
            key,
            role,
            system,
            prompt,
            schema,
            tools,
            harness,
            recursion_limit,
            **kwargs,
        )
        if key == "coordinate-0" and not saved:
            saved.append(
                (harness.state.model_copy(deep=True), harness.evidence())
            )
            raise RuntimeError(
                "interrupt between child checkpoint and ledger application"
            )
        return result

    monkeypatch.setattr(MultiAgentRunner, "_invoke", interrupt)

    async def exercise():
        async with AsyncSqliteSaver.from_conn_string(
            str(tmp_path / "child.db")
        ) as saver:
            service, model, reviewer, telemetry = service_for(
                monkeypatch, saver=saver, limits=RunLimits(model_calls=8)
            )
            state = RunState(
                incident=INCIDENT,
                investigation_start=START,
                investigation_end=END,
                limits=RunLimits(model_calls=8),
                allowed_tools=TOOL_NAMES,
                agent_mode="multi",
                verification_policy=Settings(
                    _env_file=None
                ).verification_policy,
            )
            with pytest.raises(DiagnosisError):
                await service.run_persisted(
                    state,
                    [],
                    persist=lambda *args: None,
                    is_cancelled=lambda: False,
                    resume=False,
                    graph_resume=False,
                )
            before = saved[0][0]
            assert before.model_calls == 1
            assert before.collaboration.phase == "coordinate"
            assert (
                before.collaboration.decision == 0
                and not before.collaboration.tasks
            )
            result = await service.run_persisted(
                *saved[0],
                persist=lambda *args: None,
                is_cancelled=lambda: False,
                resume=True,
                graph_resume=True,
            )
            assert result.state.status == RunStatus.SUCCEEDED
            assert (
                result.state.model_calls == model.calls + reviewer.calls == 5
            )
            assert model.calls == 4 and reviewer.calls == 1
            assert len(telemetry.calls) == 1
            assert result.state.collaboration.decision == 2

    asyncio.run(exercise())


def test_empty_local_catalogue_requires_feasible_remote_discovery(monkeypatch):
    first = True

    def premature(messages, calls):
        nonlocal first
        if first:
            first = False
            return submit(
                {
                    "status": "inconclusive",
                    "root_cause": "No retained evidence",
                    "confidence": 0,
                    "evidence_ids": [],
                }
            )
        return response(messages, calls)

    service, _, _, telemetry = service_for(monkeypatch, respond=premature)
    result = run(service)
    assert result.state.status == RunStatus.SUCCEEDED
    assert len(telemetry.calls) == 1
    assert result.state.collaboration.tasks
    assert any(
        "investigation_required" in json.dumps(e.model_dump(mode="json"))
        for e in result.state.events
    )


@pytest.mark.parametrize(
    "reader, enabled", [(True, True), (False, True), (True, False)]
)
def test_coordinator_handoff_counts_complete_escaped_request(reader, enabled):
    from langchain.agents.middleware import ModelRequest
    from langchain.agents.structured_output import ToolStrategy
    from langchain_core.messages import HumanMessage, SystemMessage
    from tracemind.agent.runtime.collaboration import CoordinatorDecision
    from tracemind.agent.workflow.context import (
        estimate_input,
        json_text,
        select_context,
    )

    originals = [
        {"id": f"E{i}", "attributes": {"body": '\\" quoted\n' * 120}}
        for i in range(1, 81)
    ]
    packet = {
        "originals": originals,
        "omitted_original_refs": ["E100"],
        "hypotheses": [{"counterevidence_refs": ["E80"]}],
        "tasks": [{"report": "Retained report"}],
        "review_feedback": ["Verify the operation"],
    }
    content = "Incident remains intact\nCoordinator context:\n" + json_text(
        packet
    )
    request = ModelRequest(
        model=ScriptedModel(respond=response),
        messages=[
            HumanMessage(content),
            HumanMessage("Application runtime budget"),
        ],
        system_message=SystemMessage("System rules"),
        response_format=ToolStrategy(CoordinatorDecision),
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_evidence",
                    "description": "Read retained originals",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        if reader
        else [],
        state={},
    )
    allowance = estimate_input(request) // 2
    selected = select_context(request, allowance, coordinator_handoff=enabled)
    assert request.messages[0].text == content
    if not enabled:
        assert not selected.fits
        return
    assert selected.fits and selected.after <= allowance < selected.before
    assert selected.after == estimate_input(selected.request)
    prefix, compact = selected.request.messages[0].text.split(
        "\nCoordinator context:\n"
    )
    assert prefix == "Incident remains intact"
    compact = json.loads(compact)
    assert compact["originals_reader_available"] is reader
    count = len(compact["originals"])
    assert 0 < count < len(originals)
    assert compact["originals"] == originals[:count]
    assert compact["omitted_original_refs"] == [
        "E100",
        *[e["id"] for e in originals[count:]],
    ]
    assert selected.omitted_previews == len(originals) - count
    for key in ["hypotheses", "tasks", "review_feedback"]:
        assert compact[key] == packet[key]
    too_small = select_context(request, 100, coordinator_handoff=True)
    assert not too_small.fits
    assert "Incident remains intact" in too_small.request.messages[0].text


@pytest.mark.parametrize(
    "reason",
    [StopReason.CONTEXT_LIMIT, StopReason.MODEL_BUDGET, StopReason.DEADLINE],
)
def test_interruption_after_review_retains_supported_findings(
    monkeypatch, reason
):
    from tracemind.agent.runtime.collaboration import MultiAgentRunner
    from tracemind.agent.verification import review_packet
    from tracemind.models.verification import ReviewSubject

    original_invoke = MultiAgentRunner._invoke

    async def interrupt(
        self,
        key,
        role,
        system,
        prompt,
        schema,
        tools,
        harness,
        recursion_limit,
        **kwargs,
    ):
        if key == "coordinate-2":
            harness.stop(reason)
        return await original_invoke(
            self,
            key,
            role,
            system,
            prompt,
            schema,
            tools,
            harness,
            recursion_limit,
            **kwargs,
        )

    monkeypatch.setattr(MultiAgentRunner, "_invoke", interrupt)

    def respond(messages, calls):
        packet = coordinator_packet(messages)
        if packet and packet["originals"]:
            value = proposal_payload(DRAFT)
            value["findings"].append(
                {
                    "text": "Database is unavailable",
                    "evidence_refs": [],
                    "knowledge_refs": [],
                }
            )
            return tool_call(
                "CoordinatorDecision",
                {
                    "action": "submit",
                    "tasks": [],
                    "hypotheses": [],
                    "reason": "Review findings",
                    "proposal": value,
                },
            )
        return response(messages, calls)

    body = review()
    body["assessments"].append(
        {
            "item_id": "F2",
            "kind": "diagnostic",
            "verdict": "insufficient",
            "missing_observations": ["Database failure"],
            "rationale": "No supporting observation",
        }
    )
    service, _, reviewer, _ = service_for(
        monkeypatch, respond=respond, reviewer=ReviewModel(body=body)
    )
    result = run(service)
    assert reviewer.calls == 1
    assert result.state.stop_reason == reason
    assert result.diagnosis.status.value == "inconclusive"
    assert "Payment Charge failed" in result.diagnosis.root_cause
    assert "Database is unavailable" not in result.diagnosis.root_cause
    assert result.diagnosis.evidence_ids == [EVIDENCE.id]
    assert not result.state.candidate_items
    assert result.state.verification.review.claims[0].verdict == "supported"
    packet = review_packet(
        ReviewSubject(
            diagnosis=result.diagnosis,
            evidence=result.evidence,
            items=result.state.candidate_items,
            start=START,
            end=END,
        )
    )
    assert "Payment Charge failed" in packet["items"][0]["text"]


def test_prior_review_cannot_publish_support_after_new_observations(
    monkeypatch,
):
    from tracemind.agent.verification import review_still_current

    service, _, _, _ = service_for(monkeypatch)
    result = run(service)
    record = result.state.verification
    assert review_still_current(result.state, record, result.evidence)
    changed = [
        *result.evidence,
        EVIDENCE.model_copy(
            update={"id": uuid4(), "summary": "Later counterevidence"}
        ),
    ]
    assert not review_still_current(result.state, record, changed)
    result.state.events[-1] = result.state.events[-1].model_copy(
        update={
            "kind": "tool.completed",
            "data": {"tool": "search_traces", "evidence_ids": []},
        }
    )
    assert not review_still_current(result.state, record, result.evidence)


def test_feedback_candidate_and_counterevidence_keep_handoff_priority():
    from tracemind.models import Diagnosis
    from tracemind.models.verification import Claim

    evidence = [
        Evidence(
            source="trace",
            service="payment",
            observed_at=END,
            summary=f"Observed operation {i}",
        )
        for i in range(35)
    ]
    state = RunState(
        incident=INCIDENT,
        allowed_tools=TOOL_NAMES,
        limits=RunLimits(),
        investigation_start=START,
        investigation_end=END,
        evidence_ids=[e.id for e in evidence],
        reference_map={f"E{i + 1}": str(e.id) for i, e in enumerate(evidence)},
        collaboration=CollaborationState(round=1),
    )
    state.collaboration.draft = Diagnosis(
        incident_id=INCIDENT.id,
        status="complete",
        root_cause="Candidate under review",
        confidence=0.5,
        evidence_ids=[evidence[-2].id],
    )
    state.collaboration.feedback = [
        Claim(
            item_id="F1",
            text="Candidate under review",
            kind="diagnostic",
            verdict="contradicted",
            conflicting_ids=[evidence[-1].id],
            rationale="Counterexample",
        )
    ]
    packet = coordinator_context(state, evidence)
    assert {e["id"] for e in packet["originals"][:2]} == {"E34", "E35"}
    assert not packet["tasks"] and not packet["hypotheses"]


def test_first_review_failure_does_not_publish_unreviewed_candidate(
    monkeypatch,
):
    reviewer = ReviewModel(body=review(), mode="unavailable")
    service, _, _, _ = service_for(monkeypatch, reviewer=reviewer)
    with pytest.raises(DiagnosisError) as caught:
        run(service)
    state = caught.value.state
    assert state.status is RunStatus.FAILED
    assert state.stop_reason is StopReason.VERIFICATION_FAILED
    assert state.diagnosis is None
    assert not state.collaboration.reviews
    assert not any(e.kind in {"diagnosis.partial", "diagnosis.accepted"}
                   for e in state.events)


@pytest.mark.parametrize("new_evidence", [False, True])
def test_later_review_outage_reuses_only_current_prior_support(
    monkeypatch, new_evidence,
):
    from tracemind.agent.verification import review_still_current

    changed = EVIDENCE.model_copy(update={
        "id": uuid4(), "summary": "Later counterevidence",
    })

    class ChangedTelemetry(FixedTelemetry):
        async def query(self, expression, *, at=None):
            if expression == "counterevidence":
                self.calls.append(expression)
                return [changed]
            return await super().query(expression, at=at)

    body = review()
    body["assessments"].append({
        "item_id": "F2", "kind": "diagnostic", "verdict": "insufficient",
        "missing_observations": ["Database failure"],
        "rationale": "No supporting observation",
    })
    reviewer = ReviewModel(body=body)
    dispatched = False

    def respond(messages, calls):
        nonlocal dispatched
        packet = coordinator_packet(messages)
        if packet and packet["review_feedback"]:
            if new_evidence and not dispatched:
                dispatched = True
                return dispatch_tasks({"tasks": [{
                    **TASK, "question": "counterevidence",
                }]})
            reviewer.mode = "unavailable"
            return submit({**DRAFT, "root_cause": "NEW_UNVERIFIED_SENTINEL"})
        if packet and packet["originals"]:
            value = proposal_payload(DRAFT)
            value["findings"].append({
                "text": "Database is unavailable", "evidence_refs": [],
                "knowledge_refs": [],
            })
            return tool_call("CoordinatorDecision", {
                "action": "submit", "tasks": [], "hypotheses": [],
                "reason": "Review findings", "proposal": value,
            })
        if dispatched and packet is None:
            if not any(isinstance(m, ToolMessage) for m in messages):
                return tool_call(arguments={"expression": "counterevidence"})
            return tool_call("InvestigationReport", {
                "findings": [{
                    "text": "Later counterevidence", "kind": "observation",
                    "evidence_refs": ["E2"],
                }], "missing_observations": [],
            })
        return response(messages, calls)

    service, _, _, _ = service_for(
        monkeypatch, respond=respond, reviewer=reviewer,
        telemetry=ChangedTelemetry(),
        limits=RunLimits(model_calls=24, tool_calls=20),
    )
    if new_evidence:
        with pytest.raises(DiagnosisError) as caught:
            run(service)
        state, evidence = caught.value.state, caught.value.evidence
        assert changed.id in {e.id for e in evidence}
        assert state.status is RunStatus.FAILED
        assert state.diagnosis is None
        assert not review_still_current(
            state, state.collaboration.reviews[0], evidence,
        )
        assert not any(e.kind == "diagnosis.partial" for e in state.events)
    else:
        result = run(service)
        state, evidence = result.state, result.evidence
        assert state.status is RunStatus.INCONCLUSIVE
        assert result.diagnosis.status.value == "inconclusive"
        assert "Payment Charge failed" in result.diagnosis.root_cause
        assert "Database is unavailable" not in result.diagnosis.root_cause
        assert "NEW_UNVERIFIED_SENTINEL" not in result.diagnosis.root_cause
        assert result.diagnosis.evidence_ids == [EVIDENCE.id]
        assert not state.candidate_items
        prior = state.collaboration.reviews[0]
        assert review_still_current(state, prior, evidence)
        assert any(
            e.kind == "diagnosis.partial"
            and e.data["review_input_sha256"] == prior.input_sha256
            for e in state.events
        )
        assert trace_issues(state) == []
    assert reviewer.calls == 2
    assert state.stop_reason is StopReason.VERIFICATION_FAILED
    assert not any(e.kind == "diagnosis.accepted" for e in state.events)
