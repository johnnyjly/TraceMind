"""Rejection feedback through the real agent graph and checkpoints."""

import asyncio
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace

from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import InMemorySaver
import pytest

from tracemind.agent.errors import DiagnosisError
from tracemind.agent.runtime.audit import trace_issues
from tracemind.agent.runtime.runner import LangChainAgentRunner
from tracemind.agent.service import DiagnosisService
from tracemind.agent.verification import DiagnosisVerifier, subject_hash
from tracemind.agent.workflow.triage import current_triage
from tracemind.api.app import app, get_run_repository
from tracemind.config import Settings
from tracemind.models import Evidence, EvidenceSource
from tracemind.models.run import RunState, RunStatus, StopReason
from tracemind.models.verification import VerificationPolicy
from replay_support import ReplayResponse, ReplayTelemetry
from test_harness import ScriptedModel, tool_call
from test_verifier import FIXTURE, ReviewModel


def proposal(*, revised=False, unknown=False, status="complete"):
    return {
        "status": status,
        "confidence": 0.5,
        "findings": [
            {
                "text": (
                    "Payment Charge failed"
                    if revised
                    else "Payment database failure caused checkout errors"
                ),
                "evidence_refs": ["E1"],
            }
        ],
        "hypotheses": [],
        "limitations": (
            [
                {
                    "text": ("The implementation cause is unknown; "
                             "no linked log was found"),
                    "evidence_refs": ["E1"],
                }
            ]
            if unknown
            else []
        ),
        "proposed_checks": [
            {
                "text": ("Inspect the payment Charge exception "
                         "and deployment diff"),
            }
        ],
    }


class RoundReview(ReviewModel):
    final_verdict: str = "supported"
    first_complete: bool = True

    async def _agenerate(self, messages, **kwargs):
        packet = json.loads(messages[1].content)
        first = self.calls == 0
        verdict = "insufficient" if first else self.final_verdict
        if first and not self.first_complete:
            verdict = "supported"
        claims = []
        for item in packet["items"]:
            claim = {
                "item_id": item["id"],
                "rationale": "Scripted semantic review",
            }
            if item["id"].startswith("P"):
                claim.update(kind="proposal", verdict="supported")
            elif item["id"].startswith("L"):
                claim.update(
                    kind="uncertainty",
                    verdict="supported",
                    missing_observations=["Implementation cause"],
                )
            else:
                claim.update(kind="diagnostic", verdict=verdict)
                if verdict == "supported":
                    claim["supporting_refs"] = ["E1"]
                else:
                    claim["missing_observations"] = [
                        "A linked payment exception"
                    ]
            claims.append(claim)
        complete = verdict == "supported" and not (
            first and not self.first_complete
        )
        self.body = {
            "assessments": claims,
            "root_cause_correctness": None,
            "rationale": "Control-flow fixture, not diagnostic accuracy",
            "completion": [
                {
                    "criterion_id": key,
                    "met": complete,
                    "item_ids": (
                        ["F1"] if key == "actionable_finding" else ["P1"]
                    ),
                    "missing_requirements": (
                        [] if complete else ["Supported finding"]
                    ),
                    "rationale": "Initial handoff criterion",
                }
                for key in packet["objective"]["criteria"]
            ],
        }
        return await super()._agenerate(messages, **kwargs)


def setup_revision(
    monkeypatch,
    mode="rewrite",
    *,
    final="supported",
    first_complete=True,
    calls=12,
    checkpointer=None,
):
    reviewer = RoundReview(final_verdict=final, first_complete=first_complete)
    monkeypatch.setattr(
        "tracemind.agent.verification.create_review_model",
        lambda settings: reviewer,
    )
    settings = Settings(
        _env_file=None, agent_max_model_calls=calls, agent_trace_directory=None
    )
    holder = {}

    async def respond(messages, number):
        if number == 1:
            return tool_call("search_traces", {"service": "checkout"})
        if number == 2:
            return tool_call(
                "DiagnosisProposal",
                proposal(
                    status="complete" if first_complete else "inconclusive"
                ),
            )
        if number == 3:
            assert "single feedback round" in str(messages)
            assert (
                "A linked payment exception" in str(messages)
                or not first_complete
            )
            if mode != "rewrite":
                return tool_call(
                    "search_logs", {"service": "payment", "text": "Charge"}
                )
        if mode == "deadline":
            holder["harness"].state.deadline_at = datetime.now(
                UTC
            ) - timedelta(seconds=1)
        return tool_call(
            "DiagnosisProposal",
            proposal(revised=True, unknown=mode == "unavailable"),
        )

    investigator = ScriptedModel(respond=respond)

    class Runner(LangChainAgentRunner):
        async def run(self, *args, harness, **kwargs):
            holder["harness"] = harness
            return await super().run(*args, harness=harness, **kwargs)

    responses = list(FIXTURE.responses)
    if mode != "rewrite":
        responses.append(
            ReplayResponse(
                method="search_logs",
                arguments={"service": "payment", "text": "Charge"},
                evidence=(
                    []
                    if mode == "unavailable"
                    else [
                        Evidence(
                            source=EvidenceSource.LOG,
                            service="payment",
                            observed_at=FIXTURE.start,
                            summary="Payment Charge exception",
                        )
                    ]
                ),
            )
        )
    telemetry = ReplayTelemetry(responses)
    service = DiagnosisService(
        Runner(investigator, checkpointer=checkpointer),
        tool_client=telemetry,
        limits=settings.run_limits,
        verification_policy=settings.verification_policy,
        verifier=DiagnosisVerifier(settings),
    )
    return service, investigator, reviewer, telemetry


def execute(service):
    return asyncio.run(
        service.diagnose_with_evidence(
            FIXTURE.case.incident,
            start=FIXTURE.start,
            end=FIXTURE.end,
        )
    )


@pytest.mark.parametrize("mode", ["rewrite", "evidence", "unavailable"])
def test_rejection_revises_or_gathers_evidence_then_rechecks(
    monkeypatch, mode
):
    service, investigator, reviewer, _ = setup_revision(monkeypatch, mode)
    run = execute(service)
    state = run.state
    assert state.status is RunStatus.SUCCEEDED
    assert state.verification.review.task_completed
    assert len(state.verification_history) == 1
    archived = state.verification_history[0]
    assert archived.review.verdict == "insufficient"
    assert subject_hash(archived.subject) == archived.input_sha256
    assert (
        archived.candidate.root_cause
        != state.verification.candidate.root_cause
    )
    assert state.verification.input_sha256 != archived.input_sha256
    assert reviewer.calls == 2
    assert investigator.calls == (3 if mode == "rewrite" else 4)
    assert state.model_calls == investigator.calls + reviewer.calls
    assert state.tool_calls.get("search_logs", 0) == (mode != "rewrite")
    packet = json.loads(reviewer.captured[-1][0][1].content)
    assert len(packet["evidence"]) == (2 if mode == "evidence" else 1)
    assert (
        "Scripted semantic review" not in reviewer.captured[-1][0][1].content
    )
    assert state.limits.duration_seconds == 600
    assert (state.deadline_at - state.started_at).total_seconds() == 600
    assert trace_issues(state) == []
    if mode == "unavailable":
        assert any(i.kind == "uncertainty" for i in run.diagnosis.items)


def test_supported_but_incomplete_handoff_also_gets_feedback(monkeypatch):
    service, _, reviewer, _ = setup_revision(monkeypatch, first_complete=False)
    run = execute(service)
    first = run.state.verification_history[0].review
    assert first.accepted and not first.task_completed
    assert reviewer.calls == 2 and run.state.status is RunStatus.SUCCEEDED


def test_second_rejection_stops_with_both_reviews_and_partial_report(
    monkeypatch,
):
    service, _, reviewer, _ = setup_revision(monkeypatch, final="insufficient")
    run = execute(service)
    assert reviewer.calls == 2
    assert run.state.stop_reason is StopReason.VERIFICATION_REJECTED
    assert len(run.state.verification_history) == 1
    assert run.diagnosis.items and run.diagnosis.review_status == "completed"
    assert run.diagnosis.status.value == "inconclusive"


def test_insufficient_budget_keeps_first_review_without_starting_revision(
    monkeypatch,
):
    service, _, reviewer, _ = setup_revision(monkeypatch, calls=5)
    run = execute(service)
    assert reviewer.calls == 1
    assert run.state.verification_history == []
    assert run.diagnosis.items
    assert any(
        e.kind == "investigation.revision_skipped" for e in run.state.events
    )


def test_new_evidence_then_deadline_preserves_report_without_stale_support(
    monkeypatch,
):
    service, _, reviewer, _ = setup_revision(monkeypatch, "deadline")
    run = execute(service)
    assert reviewer.calls == 1 and len(run.evidence) == 2
    assert run.state.stop_reason is StopReason.DEADLINE
    assert run.diagnosis.items and run.diagnosis.review_status == "timed_out"
    assert all(i.review_status == "unreviewed" for i in run.diagnosis.items)
    assert all(
        i.evidence_support == "not_assessed" for i in run.diagnosis.items
    )
    assert current_triage(run.state, run.evidence) is not None


@pytest.mark.parametrize(
    "cut", ["revision_start", "after_tool", "review_completed"]
)
def test_revision_checkpoint_resume_keeps_round_budget_and_completed_tools(
    monkeypatch, cut
):
    saver = InMemorySaver()
    service, investigator, reviewer, telemetry = setup_revision(
        monkeypatch, "evidence", checkpointer=saver
    )
    state = RunState(
        incident=FIXTURE.case.incident,
        investigation_start=FIXTURE.start,
        investigation_end=FIXTURE.end,
        limits=service._limits,
        allowed_tools=service._allowed_tools,
        verification_policy=service._verification_policy,
    )
    saved = {}

    def persist(current, evidence):
        trigger = bool(current.verification_history) and (
            (cut == "revision_start" and current.verification is None)
            or (
                cut == "after_tool"
                and current.model_calls == 5
                and current.events[-1].kind == "model.started"
            )
            or (
                cut == "review_completed"
                and current.verification
                and current.verification.status == "completed"
            )
        )
        if trigger and not saved:
            saved.update(
                state=RunState.model_validate_json(current.model_dump_json()),
                evidence=[e.model_copy(deep=True) for e in evidence],
            )
            raise RuntimeError("simulated worker loss after durable snapshot")

    async def exercise():
        with pytest.raises(DiagnosisError):
            await service.run_persisted(
                state,
                [],
                persist=persist,
                is_cancelled=lambda: False,
                resume=False,
                graph_resume=False,
            )
        restored = saved["state"]
        deadline = restored.deadline_at
        model_calls = restored.model_calls
        assert restored.graph_thread_id.endswith(":revision:1")
        checkpoint = await saver.aget_tuple(
            {
                "configurable": {"thread_id": restored.graph_thread_id},
            }
        )
        assert (checkpoint is None) is (cut == "revision_start")
        if cut == "revision_start":
            pending = current_triage(restored, saved["evidence"])
            assert pending.review_status == "pending"
            cancelled = current_triage(
                restored, saved["evidence"], cancel_requested=True,
            )
            assert cancelled.review_status == "unavailable"
            assert cancelled.review_error_code == "cancelled"
        result = await service.run_persisted(
            restored,
            saved["evidence"],
            persist=lambda *_: None,
            is_cancelled=lambda: False,
            resume=True,
            graph_resume=checkpoint is not None,
        )
        assert result.state.deadline_at == deadline
        assert result.state.model_calls >= model_calls
        assert result.state.status is RunStatus.SUCCEEDED
        assert len(result.state.verification_history) == 1
        assert result.state.tool_calls == {
            "search_traces": 1,
            "search_logs": 1,
        }
        assert reviewer.calls == 2
        assert trace_issues(result.state) == []
        return result

    asyncio.run(exercise())


def test_saved_policy_without_revision_field_keeps_original_execution_policy():
    state = RunState(
        incident=FIXTURE.case.incident,
        investigation_start=FIXTURE.start,
        investigation_end=FIXTURE.end,
        limits=Settings(_env_file=None).run_limits,
        allowed_tools=(),
        verification_policy=VerificationPolicy(),
    )
    assert state.verification_policy.revise_on_rejection
    raw = state.model_dump(mode="json")
    del raw["verification_policy"]["revise_on_rejection"]
    assert not RunState.model_validate(
        raw
    ).verification_policy.revise_on_rejection


def test_triage_api_keeps_historical_candidate_after_revision_deadline(
    monkeypatch,
):
    service, _, _, _ = setup_revision(monkeypatch, "deadline")
    run = execute(service)
    repository = SimpleNamespace(
        get=lambda _id: SimpleNamespace(
            state=run.state,
            evidence=run.evidence,
            cancel_requested=False,
        )
    )
    app.dependency_overrides[get_run_repository] = lambda: repository

    async def read():
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.get(f"/api/v1/runs/{run.state.id}/triage")

    try:
        response = asyncio.run(read())
    finally:
        app.dependency_overrides.pop(get_run_repository)
    assert response.status_code == 200
    assert response.json()["candidate_id"] == (
        run.state.verification_history[0].input_sha256
    )
    assert response.json()["diagnosis"] == run.diagnosis.model_dump(
        mode="json"
    )
