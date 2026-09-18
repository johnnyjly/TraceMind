"""Item review, reference integrity, budget and recovery regressions."""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from tracemind.agent.errors import DiagnosisCancelled, DiagnosisError
from tracemind.agent.runtime.audit import trace_issues
from tracemind.agent.runtime.runner import LangChainAgentRunner
from tracemind.agent.service import DiagnosisService
from tracemind.agent.verification import (
    DiagnosisVerifier,
    parse_review,
    review_input_bytes,
    review_messages,
    review_packet,
    subject_hash,
)
from tracemind.config import Settings
from tracemind.models.run import RunStatus, StopReason
from tracemind.models.verification import ReviewSubject, VerificationPolicy
from replay_support import ReplayFixture, ReplayTelemetry, ScriptedReplayModel

FIXTURE = ReplayFixture.load(Path("evals/replays/payment_failure.json"))
EVIDENCE = FIXTURE.responses[0].evidence[0]


def review(verdict="supported", **updates):
    claim = {
        "item_id": "F1",
        "kind": "diagnostic",
        "verdict": verdict,
        "rationale": "Retained payment observation",
    }
    if verdict == "supported":
        claim["supporting_refs"] = ["E1"]
    elif verdict == "contradicted":
        claim["conflicting_refs"] = ["E1"]
    else:
        claim["missing_observations"] = ["A linked failing operation"]
    result = {
        "assessments": [claim],
        "root_cause_correctness": None,
        "rationale": "Review of retained observations",
    }
    result.update(updates)
    return result


class ReviewModel(BaseChatModel):
    body: dict = {}
    mode: str = ""
    calls: int = 0
    captured: list = []

    @property
    def _llm_type(self):
        return "synthetic-review"

    def _generate(self, messages, **kwargs):
        raise AssertionError("async only")

    async def _agenerate(self, messages, **kwargs):
        self.calls += 1
        self.captured.append((messages, kwargs))
        if self.mode == "timeout":
            await asyncio.Future()
        if self.mode == "unavailable":
            raise httpx.ConnectError("synthetic provider outage")
        metadata = (
            {"finish_reason": "length"} if self.mode == "truncated" else {}
        )
        packet = json.loads(messages[1].content)
        body = json.loads(json.dumps(self.body))
        # Assess proposal-only items in the scripted fixture too.
        present = {a["item_id"] for a in body.get("assessments", [])}
        if self.mode != "missing":
            body.setdefault("assessments", []).extend(
                {
                    "item_id": item["id"],
                    "kind": "proposal",
                    "verdict": "supported",
                    "rationale": "Proposed check only",
                }
                for item in packet["items"]
                if item["id"].startswith("P") and item["id"] not in present
            )
        invalid = self.mode == "invalid" or (
            self.mode == "repair" and self.calls == 1
        )
        if packet.get("objective") and "completion" not in body:
            # This fixture is a routing oracle, not a semantic judge. Tests
            # of incomplete diagnoses pass explicit unmet criteria instead.
            diagnostic = [a["item_id"] for a in body.get("assessments", [])
                          if a["kind"] == "diagnostic"
                          and a["verdict"] == "supported"]
            body["completion"] = [
                {"criterion_id": key, "met": bool(diagnostic),
                 "item_ids": diagnostic,
                 "missing_requirements": [] if diagnostic else ["Evidence"],
                 "rationale": "Scripted completion assessment"}
                for key in packet["objective"]["criteria"]
            ]
        message = AIMessage(
            content="invalid json" if invalid else json.dumps(body),
            response_metadata=metadata,
            usage_metadata={
                "input_tokens": 20,
                "output_tokens": 10,
                "total_tokens": 30,
            },
            tool_calls=(
                [{"id": "bad", "name": "search_logs", "args": {}}]
                if self.mode == "tool"
                else []
            ),
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def make_service(monkeypatch, model, *, policy=None, calls=8, script=None):
    monkeypatch.setattr(
        "tracemind.agent.verification.create_review_model",
        lambda settings: model,
    )
    settings = Settings(
        _env_file=None,
        agent_trace_directory=None,
        agent_max_model_calls=calls,
        agent_retry_backoff_seconds=0,
    )
    investigator = ScriptedReplayModel(script=script or FIXTURE.script)
    service = DiagnosisService(
        LangChainAgentRunner(investigator),
        tool_client=ReplayTelemetry(FIXTURE.responses),
        limits=settings.run_limits,
        verification_policy=policy or VerificationPolicy(),
        verifier=DiagnosisVerifier(settings),
    )
    return service, investigator


def diagnose(service):
    return asyncio.run(
        service.diagnose_with_evidence(
            FIXTURE.case.incident, start=FIXTURE.start, end=FIXTURE.end
        )
    )


def seed():
    return ReviewSubject.model_validate(
        json.loads(
            Path("evals/langfuse/verifier-calibration.json").read_text()
        )[0]["input"]
    )


def test_supported_review_precedes_acceptance_and_shares_usage(monkeypatch):
    model = ReviewModel(body=review())
    service, investigator = make_service(monkeypatch, model)
    run = diagnose(service)
    assert run.diagnosis.status.value == "complete"
    assert run.state.model_calls == investigator.position + model.calls == 3
    assert run.state.usage.reported_total_tokens == 30
    assert trace_issues(run.state) == []
    assert run.state.candidate_items
    assert run.state.reference_map["E1"] == str(run.evidence[0].id)
    events = [e.kind for e in run.state.events]
    assert events.index("verification.completed") < events.index(
        "diagnosis.accepted"
    )
    assert run.state.verification.review.accepted
    assert model.captured[0][1]["response_format"]["type"] == "json_schema"
    assert model.captured[0][1]["reasoning"] == {"effort": "high"}
    packet = json.loads(model.captured[0][0][1].content)
    assert packet["items"][0]["id"] == "F1"
    assert "quote_sources" not in packet


@pytest.mark.parametrize("verdict", ["insufficient", "contradicted"])
def test_semantic_rejection_does_not_retry_or_accept(monkeypatch, verdict):
    model = ReviewModel(body=review(verdict))
    service, investigator = make_service(monkeypatch, model)
    run = diagnose(service)
    assert run.state.stop_reason is StopReason.VERIFICATION_REJECTED
    assert run.diagnosis.status.value == "inconclusive"
    assert run.state.verification.candidate.status.value == "complete"
    assert investigator.position == 2 and model.calls == 1
    assert trace_issues(run.state) == []


def test_exactly_one_format_repair_is_accounted(monkeypatch):
    model = ReviewModel(body=review(), mode="repair")
    service, _ = make_service(monkeypatch, model)
    run = diagnose(service)
    assert run.state.verification.review.accepted
    assert run.state.verification.attempts == 2
    assert len(run.state.verification.validation_errors) == 1
    assert run.state.model_calls == 4 and model.calls == 2
    assert len(model.captured[1][0]) == 4
    assert trace_issues(run.state) == []


@pytest.mark.parametrize(
    "mode,attempts,code",
    [
        ("invalid", 2, "invalid_output"),
        ("truncated", 2, "invalid_output"),
        ("tool", 2, "invalid_output"),
        ("timeout", 1, "timeout"),
        ("unavailable", 1, "model_unavailable"),
    ],
)
def test_failure_is_bounded_and_never_accepted(
    monkeypatch, mode, attempts, code
):
    model = ReviewModel(body=review(), mode=mode)
    service, _ = make_service(
        monkeypatch, model, policy=VerificationPolicy(timeout_seconds=0.05)
    )
    with pytest.raises(DiagnosisError) as caught:
        diagnose(service)
    state = caught.value.state
    assert state.verification.error_code == code
    assert state.verification.attempts == model.calls == attempts
    assert state.diagnosis is None
    assert trace_issues(state) == []


@pytest.mark.parametrize(
    "problem", ["unknown", "duplicate", "gaps", "reference"]
)
def test_proof_obligations_are_still_strict(monkeypatch, problem):
    body = review()
    claim = body["assessments"][0]
    if problem == "unknown":
        claim["supporting_refs"] = [str(uuid4())]
    elif problem == "duplicate":
        claim["supporting_refs"] *= 2
    elif problem == "gaps":
        claim["missing_observations"] = ["Missing mechanism"]
    else:
        body["root_cause_correctness"] = True
    model = ReviewModel(body=body)
    service, _ = make_service(monkeypatch, model)
    with pytest.raises(DiagnosisError) as caught:
        diagnose(service)
    assert caught.value.state.verification.error_code == "invalid_output"
    assert model.calls == 2


def test_oversize_does_not_drop_evidence_or_call_model(monkeypatch):
    model = ReviewModel(body=review())
    service, _ = make_service(
        monkeypatch, model, policy=VerificationPolicy(input_bytes=1)
    )
    with pytest.raises(DiagnosisError) as caught:
        diagnose(service)
    assert caught.value.state.verification.error_code == "input_limit"
    assert model.calls == 0
    assert len(caught.value.evidence) == 1


def test_budget_reserves_two_calls_without_extending_limit(monkeypatch):
    model = ReviewModel(body=review())
    service, _ = make_service(monkeypatch, model, calls=2)
    run = diagnose(service)
    assert run.diagnosis.status.value == "inconclusive"
    assert run.state.model_calls <= 2 and model.calls == 0


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown"])
def test_item_coverage_rejects_omission_duplication_and_invention(change):
    subject = seed()
    body = review()
    if change == "missing":
        body["assessments"] = []
    elif change == "duplicate":
        body["assessments"] *= 2
    else:
        body["assessments"][0]["item_id"] = "F999"
    with pytest.raises(ValueError):
        parse_review(AIMessage(content=json.dumps(body)), subject)


def test_whole_text_is_bound_by_program_including_synthetic_traffic():
    subject = seed()
    subject.diagnosis.root_cause += (
        "; the sampled failures are synthetic traffic"
    )
    subject.diagnosis.remediation = []
    parsed = parse_review(AIMessage(content=json.dumps(review())), subject)
    assert parsed.claims[0].text == subject.diagnosis.root_cause
    assert "synthetic traffic" in parsed.claims[0].text
    assert "text" not in review()["assessments"][0]


def test_packet_preserves_uncited_counterevidence_and_all_attributes():
    subject = seed()
    extra = subject.evidence[0].model_copy(deep=True)
    extra.id = uuid4()
    extra.attributes["unusual_detail"] = "x" * 70000
    subject.evidence.append(extra)
    packet = review_packet(subject)
    assert len(packet["evidence"]) == len(subject.evidence)
    for row, original in zip(packet["evidence"], subject.evidence):
        assert row["attributes"] == original.attributes
        assert row["summary"] == original.summary
        if original.query is not None:
            assert packet["queries"][row["query_index"]] == original.query
    assert packet["coverage"]["all_retained_observations_included"]


def test_hash_tracks_query_events_but_ignores_review_and_recovery_noise():
    subject = seed()
    before = subject_hash(subject)
    subject.execution_events.append(
        {"sequence": 99, "kind": "run.resumed", "data": {}}
    )
    assert subject_hash(subject) == before
    subject.execution_events.append(
        {
            "sequence": 100,
            "kind": "tool.completed",
            "data": {"tool": "search_logs"},
        }
    )
    assert subject_hash(subject) != before


def test_schema_is_counted_once_and_provider_finish_reason_checked():
    subject = seed()
    messages = review_messages(subject, provider="deepseek")
    from tracemind.agent.verification import review_schema

    schema_size = len(
        json.dumps(
            review_schema(review_packet(subject)),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    )
    assert review_input_bytes(messages, "deepseek") == (
        sum(len(m["content"].encode()) for m in messages) + schema_size
    )
    with pytest.raises(ValueError, match="incomplete_response"):
        parse_review(
            AIMessage(
                content=json.dumps(review()),
                response_metadata={"finish_reason": "length"},
            ),
            subject,
        )


class ForbiddenRunner:
    async def run(self, *args, **kwargs):
        raise AssertionError(
            "saved verifier phase must not rerun investigator"
        )


def persisted_run(service, state, evidence, snapshots, *, resume=False):
    return service.run_persisted(
        state,
        evidence,
        persist=lambda s, e: snapshots.append((s, e)),
        is_cancelled=lambda: False,
        resume=resume,
        graph_resume=False,
    )


@pytest.mark.parametrize("rubric_changed", [False, True])
@pytest.mark.parametrize("verdict", ["supported", "insufficient"])
def test_completed_verification_resumes_without_model_or_tool_calls(
    monkeypatch, rubric_changed, verdict
):
    model = ReviewModel(body=review(verdict))
    service, _ = make_service(monkeypatch, model)
    # Use a fresh durable state identical to the API's queued record.
    from tracemind.models.run import RunState

    state = RunState(
        incident=FIXTURE.case.incident,
        investigation_start=FIXTURE.start,
        investigation_end=FIXTURE.end,
        allowed_tools=service._allowed_tools,
        limits=service._limits,
        verification_policy=VerificationPolicy(),
    )
    snapshots = []
    asyncio.run(persisted_run(service, state, [], snapshots))
    saved, evidence = next(
        (s, e)
        for s, e in reversed(snapshots)
        if s.status is RunStatus.RUNNING
        and s.verification
        and s.verification.status == "completed"
    )
    restored = type(saved).model_validate_json(saved.model_dump_json())
    if rubric_changed:
        restored.verification.rubric_version = "older-rubric"
    service._runner = ForbiddenRunner()
    service._tool_client = ReplayTelemetry([])
    if rubric_changed:
        with pytest.raises(DiagnosisError) as caught:
            asyncio.run(
                persisted_run(service, restored, evidence, [], resume=True)
            )
        assert caught.value.state.stop_reason is StopReason.VERIFICATION_FAILED
        assert restored.verification.error_code == "rubric_changed"
        assert restored.diagnosis is None
    else:
        result = asyncio.run(
            persisted_run(service, restored, evidence, [], resume=True)
        )
        expected = "complete" if verdict == "supported" else "inconclusive"
        assert result.diagnosis.status.value == expected
    assert restored.model_calls == 3 and model.calls == 1
    assert trace_issues(restored) == []


def test_interrupted_review_does_not_repeat_on_resume(monkeypatch):
    model = ReviewModel(body=review())
    service, _ = make_service(monkeypatch, model)
    from tracemind.models.run import RunState

    state = RunState(
        incident=FIXTURE.case.incident,
        investigation_start=FIXTURE.start,
        investigation_end=FIXTURE.end,
        allowed_tools=service._allowed_tools,
        limits=service._limits,
        verification_policy=VerificationPolicy(),
    )
    snapshots = []
    asyncio.run(persisted_run(service, state, [], snapshots))
    saved, evidence = next(
        (s, e)
        for s, e in snapshots
        if s.verification
        and s.verification.status == "running"
        and s.model_calls == 3
    )
    service._runner = ForbiddenRunner()
    service._tool_client = ReplayTelemetry([])
    with pytest.raises(DiagnosisError) as caught:
        asyncio.run(persisted_run(service, saved, evidence, [], resume=True))
    assert caught.value.state.verification.error_code == "ambiguous_attempt"
    assert caught.value.state.stop_reason is StopReason.VERIFICATION_FAILED
    assert model.calls == 1 and saved.model_calls == 3
    assert trace_issues(saved) == []


def test_cancel_during_review_preserves_cancelled_state(monkeypatch):
    model = ReviewModel(body=review(), mode="timeout")
    service, _ = make_service(monkeypatch, model)

    async def cancel():
        task = asyncio.create_task(
            service.diagnose_with_evidence(
                FIXTURE.case.incident, start=FIXTURE.start, end=FIXTURE.end
            )
        )
        while model.calls == 0:
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(DiagnosisCancelled) as caught:
            await task
        return caught.value.state

    state = asyncio.run(cancel())
    assert state.status is RunStatus.CANCELLED
    assert state.verification.error_code == "cancelled"
    assert state.model_calls == 3 and model.calls == 1
    assert trace_issues(state) == []


def test_no_global_boolean_can_veto_supported_items():
    subject = seed()
    subject.diagnosis.remediation = []
    result = parse_review(AIMessage(content=json.dumps(review())), subject)
    assert result.accepted and result.uncertainty_appropriate
    bad = review(uncertainty_appropriate=False)
    with pytest.raises(ValueError):
        parse_review(AIMessage(content=json.dumps(bad)), subject)


def test_partial_result_keeps_only_retained_items_knowledge_citations():
    from tracemind.agent.verification import partial_draft
    from tracemind.models.verification import (
        Claim,
        DiagnosisReview,
        ReviewItem,
    )

    items = [
        ReviewItem(
            id="F1",
            section="finding",
            text="Observed failure",
            knowledge_refs=["knowledge-kept"],
        ),
        ReviewItem(
            id="F2",
            section="finding",
            text="Unsupported cause",
            knowledge_refs=["knowledge-rejected"],
        ),
    ]
    result = partial_draft(
        items,
        DiagnosisReview(
            claims=[
                Claim(
                    item_id="F1",
                    text=items[0].text,
                    kind="diagnostic",
                    verdict="supported",
                    supporting_ids=[EVIDENCE.id],
                    rationale="Observed",
                ),
                Claim(
                    item_id="F2",
                    text=items[1].text,
                    kind="diagnostic",
                    verdict="insufficient",
                    missing_observations=["Cause"],
                    rationale="Not established",
                ),
            ],
            uncertainty_appropriate=True,
            root_cause_correctness=None,
            rationale="Retain supported finding",
        ),
    )
    assert result.knowledge_refs == ["knowledge-kept"]
    assert result.evidence_ids == [EVIDENCE.id]
    assert "Unsupported cause" not in result.root_cause


def test_legacy_checkpoint_is_readable_but_cannot_resume(monkeypatch):
    model = ReviewModel(body=review())
    service, _ = make_service(monkeypatch, model)
    run = diagnose(service)
    raw = run.state.model_dump(mode="json")
    raw.pop("contract_version")
    for claim in raw["verification"]["review"]["claims"]:
        claim.pop("item_id")
    restored = type(run.state).model_validate(raw)
    assert restored.contract_version == "legacy"
    # Workers resume active checkpoints; terminal records remain immutable.
    restored.status = RunStatus.RUNNING
    restored.stop_reason = None
    calls = model.calls
    with pytest.raises(DiagnosisError) as caught:
        asyncio.run(
            persisted_run(service, restored, run.evidence, [], resume=True)
        )
    assert caught.value.state.stop_reason is StopReason.CONTRACT_CHANGED
    assert model.calls == calls


def test_reference_handles_survive_persistence_and_append():
    from tracemind.agent.workflow.references import register_references
    from tracemind.models.run import RunState

    state = RunState(
        incident=FIXTURE.case.incident,
        investigation_start=FIXTURE.start,
        investigation_end=FIXTURE.end,
        limits=Settings(_env_file=None).run_limits,
        allowed_tools=(),
        evidence_ids=[EVIDENCE.id],
    )
    register_references(state)
    restored = RunState.model_validate_json(state.model_dump_json())
    extra = uuid4()
    restored.evidence_ids.append(extra)
    register_references(restored)
    assert restored.reference_map == {"E1": str(EVIDENCE.id), "E2": str(extra)}


def test_structured_rendering_binds_all_items_and_rejects_added_prose():
    from tracemind.agent.workflow.validation import resolve_proposal
    from tracemind.models import Diagnosis
    from tracemind.models.result import DiagnosisProposal
    from tracemind.models.run import RunState
    from tracemind.agent.verification import subject_for

    state = RunState(
        incident=FIXTURE.case.incident,
        investigation_start=FIXTURE.start,
        investigation_end=FIXTURE.end,
        limits=Settings(_env_file=None).run_limits,
        allowed_tools=(),
        evidence_ids=[EVIDENCE.id],
    )
    proposal = DiagnosisProposal(
        status="inconclusive",
        confidence=0.4,
        findings=[{"text": "Payment Charge failed", "evidence_refs": ["E1"]}],
        hypotheses=[],
        limitations=[{"text": "Token mechanism is unverified"}],
        proposed_checks=[{"text": "Inspect validation configuration"}],
    )
    draft = resolve_proposal(proposal, state, [EVIDENCE])
    candidate = Diagnosis(
        incident_id=state.incident.id,
        **draft.model_dump(exclude={"feedback_updates"}),
    )
    subject = subject_for(state, candidate, [EVIDENCE])
    packet = review_packet(subject)
    assert [i["id"] for i in packet["items"]] == ["F1", "L1", "P1"]
    candidate.root_cause += " A hidden unsupported claim."
    with pytest.raises(ValueError, match="candidate_items_mismatch"):
        review_packet(subject)


def test_rich_reference_stays_offline_and_requires_all_components():
    from tracemind.agent.verification import held_out_reference, review_schema
    from tracemind.models.verification import ReferenceAssessment
    from tracemind.evaluation.models import EvaluationCase

    expected = EvaluationCase.load(
        Path("evals/cases/payment_failure.json")
    ).ground_truth.model_dump(mode="json")
    online = review_packet(seed())
    assert online["reference"] is None
    assert review_schema(online)["properties"]["reference_assessment"] == {
        "type": "null"
    }
    offline = review_packet(seed(), expected)
    assert (
        offline["reference"]["diagnostic_reference"]["version"]
        == "demo-rca-v2"
    )
    assert "root_cause_terms" not in offline["reference"]
    assert review_schema(offline)["properties"]["reference_assessment"] == {
        "$ref": "#/$defs/ReferenceAssessment"
    }
    components = {
        k: {"matches": True, "item_ids": ["F1"], "rationale": "Grounded"}
        for k in ReferenceAssessment.model_fields
    }
    assert ReferenceAssessment.model_validate(components).correct
    components["mechanism"]["matches"] = False
    assert not ReferenceAssessment.model_validate(components).correct
    with pytest.raises(ValueError, match="requires root_cause_service"):
        held_out_reference(
            {"diagnostic_reference": expected["diagnostic_reference"]}
        )
    assert held_out_reference({"root_cause_service": None}) is None


@pytest.mark.parametrize("kind", ["diagnostic", "process"])
@pytest.mark.parametrize("verdict", ["contradicted", "insufficient"])
def test_review_preserves_cross_list_support_and_counterevidence(
    kind, verdict,
):
    subject = seed()
    subject.diagnosis.remediation = []
    subject.execution_events = [
        {"sequence": 7, "kind": "tool.completed",
         "data": {"tool": "search_logs"}}
    ]
    body = review(verdict)
    claim = body["assessments"][0]
    claim.update(
        kind=kind, supporting_refs=["E1"], conflicting_refs=["E1"],
    )
    if kind == "process":
        claim.update(supporting_event_sequences=[7],
                     conflicting_event_sequences=[7])
    parsed = parse_review(AIMessage(content=json.dumps(body)), subject)
    assert not parsed.accepted and parsed.verdict == verdict
    assert parsed.claims[0].supporting_ids == parsed.claims[0].conflicting_ids
    assert parsed.claims[0].supporting_ids == [subject.evidence[0].id]
    if kind == "process":
        assert parsed.claims[0].supporting_event_sequences == [7]
        assert parsed.claims[0].conflicting_event_sequences == [7]


@pytest.mark.parametrize("field", [
    "supporting_refs", "conflicting_refs",
    "supporting_event_sequences", "conflicting_event_sequences",
])
@pytest.mark.parametrize("problem", ["duplicate", "unknown", "supported"])
def test_overlap_does_not_relax_other_review_obligations(field, problem):
    subject = seed()
    subject.diagnosis.remediation = []
    subject.execution_events = [
        {"sequence": 7, "kind": "tool.completed",
         "data": {"tool": "search_logs"}}
    ]
    body = review("contradicted")
    claim = body["assessments"][0]
    claim.update(supporting_refs=["E1"], conflicting_refs=["E1"],
                 supporting_event_sequences=[7],
                 conflicting_event_sequences=[7])
    event = "event_sequences" in field
    claim["kind"] = "process" if event else "diagnostic"
    if problem == "supported":
        claim["verdict"] = "supported"
    elif problem == "duplicate":
        claim[field] *= 2
    else:
        claim[field] = [999] if event else ["E999"]
    with pytest.raises(ValueError):
        parse_review(AIMessage(content=json.dumps(body)), subject)


def test_overlapping_out_of_window_evidence_is_still_rejected():
    subject = seed()
    subject.diagnosis.remediation = []
    subject.start = subject.end
    body = review("contradicted")
    body["assessments"][0]["supporting_refs"] = ["E1"]
    with pytest.raises(ValueError, match="out_of_window"):
        parse_review(AIMessage(content=json.dumps(body)), subject)


def test_online_overlap_rejection_completes_without_format_retry(monkeypatch):
    body = review("contradicted")
    body["assessments"][0]["supporting_refs"] = ["E1"]
    model = ReviewModel(body=body)
    service, _ = make_service(monkeypatch, model)
    run = diagnose(service)
    assert model.calls == 1
    assert run.state.verification.status == "completed"
    assert not run.state.verification.validation_errors
    assert run.state.verification.review.verdict == "contradicted"
    assert run.diagnosis.status.value == "inconclusive"
    assert run.state.stop_reason is StopReason.VERIFICATION_REJECTED
    assert "diagnosis.accepted" not in [e.kind for e in run.state.events]
