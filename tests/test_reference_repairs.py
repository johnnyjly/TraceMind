"""Omitted-prefix protocol repair never substitutes semantic validation."""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage

from tracemind.agent.errors import DiagnosisError
from tracemind.agent.runtime.collaboration import CoordinatorDecision
from tracemind.agent.verification import parse_review, subject_hash
from tracemind.agent.workflow.references import normalize_reference_handles
from tracemind.models.result import ProposalStatement
from test_verifier import (ReviewModel, make_service, diagnose, review, seed)


@pytest.mark.parametrize(
    "value", ["0", "01", " 1", "1 ", "１", "+1", "1.0",
              "E999", "999", "K1", "e1"],
)
def test_reference_normalizer_does_not_guess(value):
    original = ProposalStatement(
        text="E1, 1 and K1 are prose", evidence_refs=[value])
    result, repairs = normalize_reference_handles(
        original, {"E1": "e", "K1": "k"})
    assert result == original and not repairs


def test_typed_fields_are_immutable_and_idempotent():
    original = CoordinatorDecision(
        action="submit", tasks=[], reason="1 is prose",
        hypotheses=[{"id": "1", "text": "1", "kind": "hypothesis",
                     "evidence_refs": ["1"], "knowledge_refs": ["1"]}],
        proposal={"status": "inconclusive", "confidence": .5,
                  "findings": [{"text": "1", "evidence_refs": ["1"]}],
                  "hypotheses": [], "limitations": [], "proposed_checks": []},
    )
    result, repairs = normalize_reference_handles(
        original, {"E1": "e", "K1": "k"})
    assert original.hypotheses[0].evidence_refs == ["1"]
    assert result.hypotheses[0].evidence_refs == ["E1"]
    assert result.hypotheses[0].knowledge_refs == ["K1"]
    assert result.hypotheses[0].id == result.hypotheses[0].text == "1"
    assert result.proposal.findings[0].evidence_refs == ["E1"]
    assert len(repairs) == 3 and all(p["raw"] == "1" for p in repairs)
    assert normalize_reference_handles(
        result, {"E1": "e", "K1": "k"}) == (result, [])


def test_review_repair_still_rejects_duplicate_and_out_of_window_evidence():
    subject = seed()
    subject.diagnosis.remediation = []
    before = subject_hash(subject)
    body = review()
    body["assessments"][0]["supporting_refs"] = ["1"]
    raw = AIMessage(content=json.dumps(body))
    repairs = []
    parsed = parse_review(raw, subject, reference_repairs=repairs)
    assert parsed.accepted and repairs[0]["canonical"] == "E1"
    assert '"1"' in raw.text and subject_hash(subject) == before
    for refs in (["E1", "1"], ["999"], ["K1"]):
        body["assessments"][0]["supporting_refs"] = refs
        with pytest.raises(ValueError):
            parse_review(
                AIMessage(content=json.dumps(body)), subject,
                reference_repairs=[])
    body["assessments"][0]["supporting_refs"] = ["1"]
    subject.start = subject.end
    with pytest.raises(ValueError, match="out_of_window"):
        parse_review(
                AIMessage(content=json.dumps(body)), subject,
                reference_repairs=[])


def test_supported_uncertainty_requires_specific_gap_and_names_item():
    subject = seed()
    subject.diagnosis.remediation = []
    body = review()
    body["assessments"][0].update(kind="uncertainty", supporting_refs=[])
    raw = AIMessage(content=json.dumps(body))
    with pytest.raises(ValueError, match="item_id=F1"):
        parse_review(raw, subject, reference_repairs=[])
    body["assessments"][0]["missing_observations"] = [
        "The affected request destination"]
    assert parse_review(AIMessage(content=json.dumps(body)), subject).accepted


def test_online_repair_has_audit_without_paid_retry(monkeypatch):
    body = review()
    body["assessments"][0]["supporting_refs"] = ["1"]
    model = ReviewModel(body=body)
    service, _ = make_service(monkeypatch, model)
    result = diagnose(service)
    assert model.calls == 1 and result.state.verification.review.accepted
    events = [e for e in result.state.events
              if e.kind == "reference.normalized"]
    assert len(events) == 1 and events[0].data["raw"] == "1"


def test_invalid_review_keeps_normalization_audit_and_retry_bound(monkeypatch):
    body = review()
    body["assessments"][0]["supporting_refs"] = ["E1", "1"]
    model = ReviewModel(body=body)
    service, _ = make_service(monkeypatch, model)
    with pytest.raises(DiagnosisError) as caught:
        diagnose(service)
    assert model.calls == 2
    assert caught.value.state.verification.error_code == "invalid_output"
    assert len([e for e in caught.value.state.events
                if e.kind == "reference.normalized"]) == 2


def test_real_graph_normalizes_original_coordinator_and_specialist_outputs(
    monkeypatch,
):
    from test_multi_agent import response, service_for, run

    def bare_refs(messages, calls):
        message = response(messages, calls).model_copy(deep=True)
        for call in message.tool_calls:
            args = call["args"]
            if call["name"] == "InvestigationReport":
                args["findings"][0]["evidence_refs"] = ["1"]
            if (call["name"] == "CoordinatorDecision"
                    and args["action"] == "submit"):
                args["hypotheses"] = [{
                    "id": "h1", "text": "Unconfirmed alternative",
                    "kind": "hypothesis", "evidence_refs": ["1"]}]
                args["proposal"]["findings"][0]["evidence_refs"] = ["1"]
        return message

    service, _, _, _ = service_for(monkeypatch, respond=bare_refs)
    result = run(service)
    assert result.state.verification.review.accepted
    assert result.state.collaboration.hypotheses[0].evidence_refs == ["E1"]
    tasks = result.state.collaboration.tasks.values()
    assert all(t.report.findings[0].evidence_refs == ["E1"] for t in tasks)
    events = [e for e in result.state.events
              if e.kind == "reference.normalized"]
    assert {e.data["role"] for e in events} == {"coordinator", "telemetry"}
    assert len(events) == 3


@pytest.mark.parametrize("second_succeeds", [False, True])
def test_judge_keeps_attempt_audit_on_success_and_failure(
    monkeypatch, second_succeeds,
):
    from langchain_core.outputs import ChatGeneration, ChatResult
    from tracemind.config import Settings
    from tracemind.evaluation.budget import ModelCallBudget
    from tracemind.evaluation.judge import diagnosis_judge
    from test_langfuse import FakeModel, saved, valid_review

    class Model(FakeModel):
        async def _agenerate(self, messages, **kwargs):
            self.entered.append(messages)
            body = valid_review(saved())
            body["assessments"][0]["supporting_refs"] = (
                ["1"] if second_succeeds and len(self.entered) == 2
                else ["E1", "1"]
            )
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content=json.dumps(body)))])

    model = Model()
    monkeypatch.setattr(
        "tracemind.evaluation.judge.create_review_model", lambda _: model)
    monkeypatch.setattr(
        "tracemind.evaluation.judge.callbacks_for", lambda _: [])
    evaluate = diagnosis_judge(
        Settings(_env_file=None), None, ModelCallBudget(2))
    scores = asyncio.run(evaluate(output=saved()))
    completed = next(s for s in scores if s.name == "judge_completed")
    assert completed.value is second_succeeds
    assert [r["attempt"] for r
            in completed.metadata["reference_repairs"]] == [1, 2]
    assert len(model.entered) == 2


def test_single_agent_protocol_repair_preserves_raw_response(monkeypatch):
    from replay_support import ScriptedReplayModel

    original = ScriptedReplayModel._agenerate
    returned = []

    async def bare_proposal(self, messages, **kwargs):
        result = await original(self, messages, **kwargs)
        message = result.generations[0].message
        for call in message.tool_calls:
            if call["name"] == "DiagnosisProposal":
                call["args"]["findings"][0]["evidence_refs"] = ["1"]
                returned.append(message)
        return result

    monkeypatch.setattr(ScriptedReplayModel, "_agenerate", bare_proposal)
    service, investigator = make_service(
        monkeypatch, ReviewModel(body=review()))
    result = diagnose(service)
    assert (result.state.verification.review.accepted
            and investigator.position == 2)
    assert (returned[0].tool_calls[0]["args"]["findings"][0]["evidence_refs"]
            == ["1"])
    events = [e for e in result.state.events
              if e.kind == "reference.normalized"]
    assert len(events) == 1 and events[0].data["role"] == "diagnosis"
