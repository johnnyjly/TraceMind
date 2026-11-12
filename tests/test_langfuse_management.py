"""Managed prompt identity, calibrated-label provenance and run metrics."""

import asyncio
from datetime import timedelta
from hashlib import sha256
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tracemind.config import Settings
from tracemind.agent.verification import (
    subject_hash, review_messages, pin_review_prompt, VERSION,
)
from tracemind.models.prompts import PromptSnapshot
from tracemind.models.run import RunState, RunLimits, RunEvent, RunStatus, StopReason
from tracemind.observability.prompts import pin_prompt, INVESTIGATOR_CONTRACT
from tracemind.observability.metrics import estimate_cost, run_metrics, publish_run_metrics
from tracemind.evaluation.annotations import annotation_dataset, frozen_subject
from tracemind.evaluation.evaluators import verification_scores
from tracemind.evaluation.models import ExperimentOutput
from test_verifier import seed, make_service, ReviewModel, review, diagnose
from test_langfuse import sdk


def state_for(subject=None):
    subject = subject or seed()
    return RunState(incident=subject.incident, investigation_start=subject.start,
                    investigation_end=subject.end, limits=RunLimits(), allowed_tools=())


class Prompts:
    version = 1
    calls = 0
    contract = INVESTIGATOR_CONTRACT

    def get_prompt(self, name, **kwargs):
        self.calls += 1
        return SimpleNamespace(name=name, version=self.version,
                               prompt="instructions v" + str(self.version),
                               config={"contract_version": self.contract})


def test_prompt_freezes_across_serialization_revision_and_label_change():
    state, client = state_for(), Prompts()
    async def pin(s):
        return await pin_prompt(s, "investigator", client=client,
                                label="production", contract=INVESTIGATOR_CONTRACT)
    first = asyncio.run(pin(state))
    client.version = 2
    restored = RunState.model_validate_json(state.model_dump_json())
    restored.model_calls = 3
    assert asyncio.run(pin(restored)) == first
    assert client.calls == 1
    assert asyncio.run(pin(state_for())).version == 2
    with pytest.raises(ValueError, match="contract changed"):
        asyncio.run(pin_prompt(restored, "investigator", client=client,
                               label="production", contract="new-contract"))
    corrupt = first.model_dump(); corrupt["content"] += "changed"
    with pytest.raises(ValidationError, match="hash mismatch"):
        PromptSnapshot.model_validate(corrupt)


def test_legacy_run_does_not_fetch_a_new_prompt():
    state, client = state_for(), Prompts()
    state.model_calls = 1
    assert asyncio.run(pin_prompt(state, "investigator", client=client,
                                  label="production", contract=INVESTIGATOR_CONTRACT)) is None
    assert client.calls == 0


def test_verifier_snapshot_changes_review_identity_and_preserves_text():
    subject, client = seed(), Prompts()
    client.contract = VERSION
    state = state_for(subject)
    snapshot = asyncio.run(pin_review_prompt(
        state, subject, Settings(_env_file=None, langfuse_prompt_label="production"), client))
    managed = subject.model_copy(update={"prompt": snapshot})
    assert subject_hash(managed) != subject_hash(subject)
    assert review_messages(managed)[0]["content"] == snapshot.content


@pytest.mark.parametrize("followup", [False, True])
def test_managed_service_persists_both_roles_and_exact_candidate(monkeypatch, sdk, followup):
    client, exporter = sdk
    from tracemind.agent.runtime.runner import SYSTEM_PROMPT
    from tracemind.agent.verification import RUBRIC
    calls = []
    def get_prompt(name, **kwargs):
        calls.append(name)
        investigator = name.endswith("investigator")
        return SimpleNamespace(name=name, version=3,
            prompt=SYSTEM_PROMPT if investigator else RUBRIC,
            config={"contract_version": INVESTIGATOR_CONTRACT if investigator else VERSION})
    monkeypatch.setattr(client, "get_prompt", get_prompt)
    service, _ = make_service(monkeypatch, ReviewModel(body=review()))
    settings = Settings(_env_file=None, langfuse_prompt_label="production")
    service._settings = settings
    service._langfuse = client
    service._verifier.client = client
    service._verifier.settings = settings
    if followup:
        from test_verifier import FIXTURE
        state = RunState(
            incident=FIXTURE.case.incident,
            investigation_start=FIXTURE.start, investigation_end=FIXTURE.end,
            limits=service._limits, allowed_tools=service._allowed_tools,
            verification_policy=service._verification_policy,
            prompt_version="single-agent-followup-v10-evidence-timing",
        )
        run = asyncio.run(service.run_persisted(
            state, [], persist=lambda *args: None, is_cancelled=lambda: False,
            resume=False, graph_resume=False,
        ))
    else:
        run = diagnose(service)
    assert set(run.state.prompts) == {"investigator", "verifier"}
    assert len(calls) == 2
    record = run.state.verification
    assert record.subject.prompt.version == 3
    assert frozen_subject(run.state, run.evidence, record) == record.subject
    client.flush()


def event(state, kind, seconds=0, **data):
    return RunEvent(run_id=state.id, sequence=len(state.events)+1,
                    kind=kind, span_id=state.id,
                    timestamp=state.started_at+timedelta(seconds=seconds), data=data)


def test_cost_counts_reasoning_once_and_keeps_missing_calls_unknown():
    state = state_for()
    events = [event(state, "model.started"), event(state, "model.completed",
        model="deepseek-flash", usage_reported=True, cache_usage_reported=True,
        input_tokens=100, cache_hit_tokens=20, cache_miss_tokens=80,
        output_tokens=50, reasoning_tokens=40)]
    result = estimate_cost(events)
    assert result["total_cost_estimate_usd"] == pytest.approx((20*.006+80*.3+50*1.2)/1e6)
    events.append(event(state,"model.started"))
    result = estimate_cost(events)
    assert result["total_cost_estimate_usd"] is None
    assert result["unpriced_calls"] == 1 and result["cost_coverage"] == .5


def test_wall_time_and_delivery_without_verifier_and_stable_metric_ids():
    state = state_for(); state.finished_at = state.started_at+timedelta(seconds=120)
    state.status, state.stop_reason = RunStatus.SUCCEEDED, StopReason.COMPLETED
    state.events = [event(state, "diagnosis.delivered", 100)]
    summary = run_metrics(state)
    assert summary["scores"]["first_triage_ms"] == 100000
    assert summary["scores"]["final_ms"] == 120000
    assert summary["scores"]["review_completed"] is None
    calls=[]; client=SimpleNamespace(create_score=lambda **kw: calls.append(kw))
    publish_run_metrics(client,state); first={c["score_id"] for c in calls}
    calls.clear(); publish_run_metrics(client,state)
    assert first == {c["score_id"] for c in calls}


def annotation():
    subject = seed()
    entry={"queue_id":"q", "queue_item_id":"qi", "observation_id":"obs",
           "trace_id":"trace", "incident_family":"cart", "split":"development",
           "input_sha256":subject_hash(subject), "input":subject.model_dump(mode="json")}
    score={"id":"s", "name":"tracemind.human.verdict", "value":"supported",
           "source":"ANNOTATION", "author_user_id":"human", "queue_id":"q",
           "subject":{"kind":"observation","id":"obs","trace_id":"trace"}}
    return entry, score


def test_human_labels_roundtrip_without_leakage_and_duplicates():
    entry, score = annotation()
    result=annotation_dataset([entry,entry],{("q","obs")},[score])
    assert result["candidate_count"] == result["labelled_count"] == 1
    item=result["datasets"]["development"][0]
    assert item["expected_output"] == {"verdict":"supported"}
    assert "author_user_ids" not in item["input"]
    assert item["metadata"]["source_score_ids"] == ["s"]


@pytest.mark.parametrize("mutation",["prefill","wrong_trace","wrong_queue","unknown","conflict","pending"])
def test_unproven_or_conflicting_labels_do_not_enter_dataset(mutation):
    entry, score=annotation(); completed={("q","obs")}; scores=[score]
    if mutation=="prefill": score["author_user_id"]=None
    elif mutation=="wrong_trace": score["subject"]["trace_id"]="other"
    elif mutation=="wrong_queue": score["queue_id"]="other"
    elif mutation=="unknown": score["value"]="unknown"
    elif mutation=="conflict": scores.append({**score,"id":"s2","value":"insufficient"})
    elif mutation=="pending": completed={("other","obs")}
    result=annotation_dataset([entry],completed,scores)
    assert result["labelled_count"] == 0 and sum(result["skipped"].values()) == 1


def test_same_incident_family_cannot_cross_splits():
    entry,score=annotation()
    with pytest.raises(ValueError,match="multiple dataset splits"):
        annotation_dataset([entry,{**entry,"observation_id":"other","split":"holdout"}],set(),[])


def test_unfinished_review_never_counts_as_correct_rejection():
    state=state_for()
    observed=ExperimentOutput(run_state=state)
    scores={e.name:e.value for e in verification_scores(
        output=observed.model_dump(mode="json"),expected_output={"verdict":"insufficient"})}
    assert scores["review_completed"] is False
    assert "false_acceptance" not in scores and "false_rejection" not in scores
