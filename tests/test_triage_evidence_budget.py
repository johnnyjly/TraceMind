"""Regression contracts from cart, catalog and payment live investigations."""

import asyncio
from datetime import timedelta
import json
from uuid import uuid4

import pytest

from tracemind.agent.runtime.harness import RunHarness
from tracemind.agent.runtime.tools import EvidenceCollector, create_local_tools
from tracemind.agent.verification import review_messages, review_packet
from tracemind.agent.workflow.context import _evidence_preview, intake_context
from tracemind.agent.workflow.observations import mark_observation_window
from tracemind.agent.workflow.references import register_references
from tracemind.models import Incident
from tracemind.models.run import RunLimits, RunState, RunStatus
from tracemind.models.verification import (
    VerificationPolicy, VerificationRecord,
)
from test_initial_evidence import START, END, context, observation
from test_verifier import seed


def run_state(**changes):
    values = dict(incident=Incident(title="checkout", description="failure"),
                  investigation_start=START, investigation_end=END,
                  limits=RunLimits(), allowed_tools=("read_evidence",))
    return RunState(**(values | changes))


def test_cross_window_status_and_in_window_exception_are_independent():
    item = observation(attributes={
        "timestamp_source": "span_startTime", "duration_ms": 700_000,
        "is_error": True, "exceptions": [
            {"timestamp": (END + timedelta(seconds=1)).isoformat()},
            {"timestamp": (START + timedelta(seconds=11)).isoformat()},
            {"message": "unknown time"},
        ],
    })
    marked = mark_observation_window(item, START, END)
    assert marked.attributes["span_window"] == {
        "end": (item.observed_at
                + timedelta(milliseconds=700_000)).isoformat(),
        "end_in_investigation_window": False,
    }
    assert [e["in_investigation_window"]
            for e in marked.attributes["exceptions"]] == [False, True, None]
    assert "span_window" not in item.attributes
    changed = mark_observation_window(marked, START, END + timedelta(days=1))
    assert changed.attributes["span_window"]["end_in_investigation_window"]
    assert not marked.attributes["span_window"]["end_in_investigation_window"]


@pytest.mark.parametrize("duration", [
    None, -1, True, "10", float("inf"), 1e99, 10**400,
])
def test_unknown_completion_is_not_invented(duration):
    item = observation(attributes={
        "timestamp_source": "span_startTime", "duration_ms": duration,
    })
    marked = mark_observation_window(item, START, END)
    assert marked.attributes["span_window"] == {
        "end": None, "end_in_investigation_window": None,
    }


def test_completion_boundary_and_timestamp_provenance():
    item = observation(attributes={
        "timestamp_source": "span_startTime", "duration_ms": 590_000,
    })
    assert mark_observation_window(item, START, END).attributes[
        "span_window"]["end_in_investigation_window"]
    item.attributes.pop("timestamp_source")
    item.attributes["span_window"] = {"end": "untrusted derived field"}
    marked = mark_observation_window(item, START, END)
    assert "span_window" not in marked.attributes


def test_control_trace_business_fields_survive_bounded_initial_context():
    error = observation(service="cart", attributes={
        "trace_id": "a" * 32, "span_id": "1" * 16,
        "timestamp_source": "span_startTime", "duration_ms": 700_000,
        "operation": "EmptyCart", "is_error": True,
        "diagnostic": {"span.kind": "server", "grpc.status_code": 9,
                       "user.id.sha256": "f" * 64},
        "exceptions": [{"message": "storage", "stacktrace": "x" * 8000}],
    })
    controls = [observation(service="cart", attributes={
        "trace_id": "b" * 32, "span_id": "2" * 16,
        "timestamp_source": "span_startTime", "duration_ms": 1,
        "operation": "GetCart", "is_error": False,
        "diagnostic": {"span.kind": "server", "grpc.status_code": 0,
                       "user.id.sha256": "f" * 64, "demo.cart.items.count": 1},
        "large": "x" * 8000,
    })]
    evidence = [error, *[observation(attributes={
        "trace_id": "a" * 32, "operation": "Other", "duration_ms": 1,
    }) for _ in range(35)], *controls]
    evidence = [mark_observation_window(e, START, END) for e in evidence]
    state = run_state(initial_context=context(),
                      evidence_ids=[e.id for e in evidence])
    register_references(state)
    packet = json.loads(intake_context(state, evidence))
    first, control = packet["evidence_previews"][:2]
    assert first["attributes"]["operation"] == "EmptyCart"
    assert not first["attributes"]["span_window"]["end_in_investigation_window"]
    assert control["attributes"]["diagnostic"]["demo.cart.items.count"] == 1
    assert control["attributes"]["diagnostic"]["user.id.sha256"] == "f" * 64
    for e in (evidence[0], evidence[-1]):
        assert len(json.dumps(_evidence_preview(e, max_bytes=1200),
                              separators=(",", ":")).encode()) <= 1200


def test_frozen_review_gets_timing_without_context_or_subject_mutation():
    subject = seed()
    subject.initial_context = None
    subject.evidence = [observation(attributes={
        "timestamp_source": "span_startTime", "duration_ms": 700_000,
        "is_error": True,
    })]
    subject.start, subject.end = START, END
    subject.diagnosis.evidence_ids = [subject.evidence[0].id]
    original = subject.model_dump_json()
    packet = review_packet(subject)
    assert not packet["evidence"][0]["attributes"]["span_window"][
        "end_in_investigation_window"]
    rubric = review_messages(subject)[0]["content"]
    assert "not a failure already established inside the window" in rubric
    assert subject.model_dump_json() == original


def test_first_local_page_inspection_counts_once_across_resume():
    async def scenario():
        evidence = [observation(summary=str(i), attributes={})
                    for i in range(45)]
        state = run_state(status=RunStatus.RUNNING,
                         limits=RunLimits(no_progress_rounds=1),
                         allowed_tools=("get_trace_detail", "read_evidence"),
                         evidence_ids=[e.id for e in evidence])
        register_references(state)
        collector = EvidenceCollector(initial=evidence)
        harness = RunHarness(state)
        async def prefetched():
            return [collector.preview(str(e.id)) for e in evidence]
        await harness.tool_call("get_trace_detail", {}, prefetched)
        harness.check_progress()
        reader = create_local_tools(collector, harness)[0]
        for offset in (0, 20, 40):
            await reader.ainvoke({"offset": offset})
            harness.check_progress()
            assert harness.finalization_reason is None
        assert len(collector.evidence) == 45
        saved = RunState.model_validate_json(state.model_dump_json())
        resumed = RunHarness(saved, resume=True)
        reader = create_local_tools(collector, resumed)[0]
        before = resumed.state.progress_count
        await reader.ainvoke({"offset": 40, "service": "payment"})
        assert resumed.state.progress_count == before
        resumed.check_progress()
        assert resumed.finalization_reason.value == "no_progress"
    asyncio.run(scenario())


def test_revision_phase_shares_call_token_and_deadline_reserves_after_resume():
    state = run_state(
        status=RunStatus.RUNNING, verification_policy=VerificationPolicy(),
        limits=RunLimits(model_calls=12, total_tokens=1_000_000),
    )
    harness = RunHarness(state)
    assert harness.finalization_calls == 7
    assert 12 - harness.finalization_calls == 5
    record = VerificationRecord(
        candidate=seed().diagnosis, input_sha256="saved", rubric_version="test",
    )
    state.verification_history.append(record)
    state.model_calls = 9
    harness.start_revision()
    assert harness.finalization_calls == 2
    assert (state.limits.model_calls - state.model_calls
            - harness.finalization_calls) == 1
    state.diagnosis_repairs = 1
    saved = RunState.model_validate_json(state.model_dump_json())
    restored = RunHarness(saved, resume=True)
    assert restored.state.deadline_at == state.deadline_at
    assert restored.state.model_calls == 9
    assert restored.state.diagnosis_repairs == 1
    assert restored.investigation_token_pool == (
        1_000_000 - state.limits.context_tokens
        - state.verification_policy.context_tokens)
    assert restored.verification_calls == 1
    restored.state.model_calls += 1
    assert (restored.state.limits.model_calls - restored.state.model_calls
            == restored.finalization_calls)


def test_five_explorations_both_repairs_and_revision_fit_twelve(monkeypatch):
    from tracemind.agent.runtime.runner import LangChainAgentRunner
    from tracemind.agent.service import DiagnosisService
    from tracemind.agent.verification import DiagnosisVerifier
    from tracemind.config import Settings
    from replay_support import ReplayResponse, ReplayTelemetry
    from test_harness import ScriptedModel, tool_call
    from test_revision import RoundReview, proposal, execute
    from test_verifier import FIXTURE

    class RepairThenReject(RoundReview):
        async def _agenerate(self, messages, **kwargs):
            self.final_verdict = (
                "insufficient" if self.calls < 2 else "supported"
            )
            result = await super()._agenerate(messages, **kwargs)
            if self.calls == 1:
                result.generations[0].message.content = "invalid review JSON"
            return result

    reviewer = RepairThenReject()
    monkeypatch.setattr("tracemind.agent.verification.create_review_model",
                        lambda settings: reviewer)
    settings = Settings(_env_file=None, agent_trace_directory=None,
                        agent_max_model_calls=12)

    def respond(messages, count):
        if count <= 5:
            assert "Investigation is closed" not in messages[-1].content
            return tool_call("search_logs", {
                "service": "payment", "text": str(count),
            })
        if count in (6, 7):
            assert "Investigation is closed" in messages[-1].content
            return tool_call("DiagnosisProposal", proposal(
                status="invalid" if count == 6 else "complete",
            ))
        if count == 8:
            assert "single feedback round" in str(messages)
            assert "Investigation is closed" not in messages[-1].content
            return tool_call("search_logs", {
                "service": "payment", "text": "revision",
            })
        return tool_call("DiagnosisProposal", proposal(revised=True))

    investigator = ScriptedModel(respond=respond)
    responses = [ReplayResponse(method="search_logs", arguments={
        "service": "payment", "text": key,
    }, evidence=[FIXTURE.responses[0].evidence[0].model_copy(update={
        "id": uuid4(), "summary": "Payment failed " + key,
    })]) for key in ["1", "2", "3", "4", "5", "revision"]]
    service = DiagnosisService(
        LangChainAgentRunner(investigator),
        tool_client=ReplayTelemetry(responses),
        limits=settings.run_limits,
        verification_policy=settings.verification_policy,
        verifier=DiagnosisVerifier(settings),
    )
    run = execute(service)
    assert run.state.status == RunStatus.SUCCEEDED
    assert run.state.model_calls == 12
    assert run.state.diagnosis_repairs == 1
    assert reviewer.calls == 3
    assert run.state.verification_history[0].attempts == 2
    assert run.state.verification.review.accepted
    assert run.state.tool_calls["search_logs"] == 6
    assert (run.state.deadline_at
            - run.state.started_at).total_seconds() == 600
