"""Online verification, shared admission and durable recovery boundaries."""

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
from tracemind.agent.verification import DiagnosisVerifier
from tracemind.config import Settings
from tracemind.models.run import RunStatus, StopReason
from tracemind.models.verification import VerificationPolicy
from replay_support import ReplayFixture, ReplayTelemetry, ScriptedReplayModel

FIXTURE = ReplayFixture.load(Path('evals/replays/payment_failure.json'))
EVIDENCE = FIXTURE.responses[0].evidence[0]


def review(verdict='supported', **updates):
    claim = {'text': 'Payment Charge failed', 'verdict': verdict,
             'rationale': 'Retained payment observation'}
    if verdict == 'supported':
        claim['supporting_ids'] = [str(EVIDENCE.id)]
    elif verdict == 'contradicted':
        claim['conflicting_ids'] = [str(EVIDENCE.id)]
    else:
        claim['missing_observations'] = ['A linked failing operation']
    result = {'claims': [claim], 'uncertainty_appropriate': True,
              'root_cause_correctness': None, 'process_quality': None,
              'rationale': 'Review of retained observations'}
    result.update(updates)
    return result


class ReviewModel(BaseChatModel):
    body: dict = {}
    mode: str = ''
    calls: int = 0
    captured: list = []

    @property
    def _llm_type(self):
        return 'synthetic-review'

    def _generate(self, messages, **kwargs):
        raise AssertionError('async only')

    async def _agenerate(self, messages, **kwargs):
        self.calls += 1
        self.captured.append((messages, kwargs))
        if self.mode == 'timeout':
            await asyncio.Future()
        if self.mode == 'unavailable':
            raise httpx.ConnectError('synthetic provider outage')
        metadata = {
            'finish_reason': 'length'} if self.mode == 'truncated' else {}
        message = AIMessage(
            content='invalid json' if self.mode == 'invalid' else json.dumps(
                self.body),
            response_metadata=metadata,
            usage_metadata={
                'input_tokens': 20,
                'output_tokens': 10,
                'total_tokens': 30},
            tool_calls=([{'id': 'bad', 'name': 'search_logs', 'args': {}}]
                        if self.mode == 'tool' else []),
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def make_service(monkeypatch, model, *, policy=None, calls=8, script=None):
    monkeypatch.setattr(
        'tracemind.agent.verification.create_chat_model',
        lambda settings: model)
    settings = Settings(_env_file=None, agent_trace_directory=None,
                        agent_max_model_calls=calls,
                        agent_retry_backoff_seconds=0)
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
    return asyncio.run(service.diagnose_with_evidence(
        FIXTURE.case.incident, start=FIXTURE.start, end=FIXTURE.end))


def test_supported_review_precedes_acceptance_and_shares_usage(monkeypatch):
    model = ReviewModel(body=review())
    service, investigator = make_service(monkeypatch, model)
    run = diagnose(service)
    assert run.diagnosis.status.value == 'complete'
    assert run.state.status is RunStatus.SUCCEEDED
    assert run.state.model_calls == investigator.position + model.calls == 3
    assert run.state.usage.reported_total_tokens == 30
    assert run.state.usage.unreported_model_calls == 2
    assert trace_issues(run.state) == []
    events = [e.kind for e in run.state.events]
    assert events.index('verification.completed') < events.index(
        'diagnosis.accepted')
    assert run.state.verification.review.accepted
    assert model.captured[0][1]['max_tokens'] == 4096
    assert model.captured[0][1]['response_format'] == {'type': 'json_object'}


@pytest.mark.parametrize('verdict', ['insufficient', 'contradicted'])
def test_unsupported_review_blocks_complete_without_repair(
        monkeypatch, verdict):
    model = ReviewModel(body=review(verdict))
    service, investigator = make_service(monkeypatch, model)
    run = diagnose(service)
    assert run.state.status is RunStatus.INCONCLUSIVE
    assert run.state.stop_reason is StopReason.VERIFICATION_REJECTED
    assert run.diagnosis.status.value == 'inconclusive'
    assert run.diagnosis.confidence == 0
    assert run.state.verification.candidate.status.value == 'complete'
    assert run.state.verification.review.verdict == verdict
    assert run.state.diagnosis_repairs == 0
    assert investigator.position == 2 and model.calls == 1
    assert 'diagnosis.accepted' not in [e.kind for e in run.state.events]
    assert trace_issues(run.state) == []


@pytest.mark.parametrize('mode', ['invalid',
                         'truncated', 'tool', 'timeout', 'unavailable'])
def test_review_failure_has_no_complete_result_or_retry(monkeypatch, mode):
    model = ReviewModel(body=review(), mode=mode)
    service, investigator = make_service(
        monkeypatch, model, policy=VerificationPolicy(timeout_seconds=.01))
    with pytest.raises(DiagnosisError) as caught:
        diagnose(service)
    state = caught.value.state
    assert state.status is RunStatus.FAILED
    assert state.stop_reason is StopReason.VERIFICATION_FAILED
    assert state.diagnosis is None
    assert state.verification.status == 'failed'
    assert state.model_calls == 3
    assert model.calls == 1 and investigator.position == 2
    assert not any(e.kind == 'recovery.retry_scheduled' for e in state.events)
    assert trace_issues(state) == []


@pytest.mark.parametrize('problem',
                         ['unknown', 'duplicate', 'gaps', 'reference'])
def test_invalid_proof_obligations_fail_closed(monkeypatch, problem):
    body = review()
    claim = body['claims'][0]
    if problem == 'unknown':
        claim['supporting_ids'] = [str(uuid4())]
    elif problem == 'duplicate':
        claim['supporting_ids'] *= 2
    elif problem == 'gaps':
        claim['missing_observations'] = ['Unresolved telemetry']
    else:
        body['root_cause_correctness'] = True
    model = ReviewModel(body=body)
    service, _ = make_service(monkeypatch, model)
    with pytest.raises(DiagnosisError) as caught:
        diagnose(service)
    assert caught.value.state.verification.error_code == 'invalid_review'
    assert caught.value.state.stop_reason is StopReason.VERIFICATION_FAILED


def test_oversized_subject_fails_before_call_without_dropping_evidence(
        monkeypatch):
    model = ReviewModel(body=review())
    service, _ = make_service(
        monkeypatch, model, policy=VerificationPolicy(
            input_bytes=1))
    with pytest.raises(DiagnosisError) as caught:
        diagnose(service)
    state = caught.value.state
    assert state.verification.error_code == 'input_limit'
    assert state.model_calls == 2 and model.calls == 0
    assert len(caught.value.evidence) == 1


def test_budget_is_not_extended_to_pay_for_verification(monkeypatch):
    model = ReviewModel(body=review())
    service, _ = make_service(monkeypatch, model, calls=2)
    run = diagnose(service)
    assert run.diagnosis.status.value == 'inconclusive'
    assert run.state.model_calls <= 2
    assert model.calls == 0


def test_inconclusive_answer_does_not_spend_a_verifier_call(monkeypatch):
    model = ReviewModel(body=review())
    script = [{'name': 'DiagnosisDraft', 'args': {
        'status': 'inconclusive', 'root_cause': 'No fault observations',
        'confidence': 0, 'cite_indices': []}}]
    service, _ = make_service(monkeypatch, model, script=script)
    run = diagnose(service)
    assert run.diagnosis.status.value == 'inconclusive'
    assert run.state.verification is None and model.calls == 0


class ForbiddenRunner:
    async def run(self, *args, **kwargs):
        raise AssertionError(
            'saved verifier phase must not rerun investigator')


def persisted_run(service, state, evidence, snapshots, *, resume=False):
    return service.run_persisted(
        state, evidence, persist=lambda s, e: snapshots.append((s, e)),
        is_cancelled=lambda: False, resume=resume, graph_resume=False)


@pytest.mark.parametrize("rubric_changed", [False, True])
def test_completed_verification_resumes_without_model_or_tool_calls(
        monkeypatch, rubric_changed):
    model = ReviewModel(body=review())
    service, _ = make_service(monkeypatch, model)
    # Use a fresh durable state identical to the API's queued record.
    from tracemind.models.run import RunState
    state = RunState(incident=FIXTURE.case.incident,
                     investigation_start=FIXTURE.start,
                     investigation_end=FIXTURE.end,
                     allowed_tools=service._allowed_tools,
                     limits=service._limits,
                     verification_policy=VerificationPolicy())
    snapshots = []
    asyncio.run(persisted_run(service, state, [], snapshots))
    saved, evidence = next((s, e) for s, e in reversed(snapshots)
                           if s.status is RunStatus.RUNNING
                           and s.verification
                           and s.verification.status == 'completed')
    restored = type(saved).model_validate_json(saved.model_dump_json())
    if rubric_changed:
        restored.verification.rubric_version = "older-rubric"
    service._runner = ForbiddenRunner()
    service._tool_client = ReplayTelemetry([])
    if rubric_changed:
        with pytest.raises(DiagnosisError) as caught:
            asyncio.run(persisted_run(
                service, restored, evidence, [], resume=True))
        assert caught.value.state.stop_reason is StopReason.VERIFICATION_FAILED
        assert restored.verification.error_code == "rubric_changed"
        assert restored.diagnosis is None
    else:
        result = asyncio.run(persisted_run(
            service, restored, evidence, [], resume=True))
        assert result.diagnosis.status.value == 'complete'
    assert restored.model_calls == 3 and model.calls == 1
    assert trace_issues(restored) == []


def test_interrupted_review_does_not_repeat_on_resume(monkeypatch):
    model = ReviewModel(body=review())
    service, _ = make_service(monkeypatch, model)
    from tracemind.models.run import RunState
    state = RunState(incident=FIXTURE.case.incident,
                     investigation_start=FIXTURE.start,
                     investigation_end=FIXTURE.end,
                     allowed_tools=service._allowed_tools,
                     limits=service._limits,
                     verification_policy=VerificationPolicy())
    snapshots = []
    asyncio.run(persisted_run(service, state, [], snapshots))
    saved, evidence = next((s, e) for s, e in snapshots
                           if s.verification
                           and s.verification.status == 'running'
                           and s.model_calls == 3)
    service._runner = ForbiddenRunner()
    service._tool_client = ReplayTelemetry([])
    with pytest.raises(DiagnosisError) as caught:
        asyncio.run(persisted_run(service, saved, evidence, [], resume=True))
    assert caught.value.state.verification.error_code == 'ambiguous_attempt'
    assert caught.value.state.stop_reason is StopReason.VERIFICATION_FAILED
    assert model.calls == 1 and saved.model_calls == 3
    assert trace_issues(saved) == []


def test_cancel_during_review_preserves_cancelled_state(monkeypatch):
    model = ReviewModel(body=review(), mode='timeout')
    service, _ = make_service(monkeypatch, model)

    async def cancel():
        task = asyncio.create_task(service.diagnose_with_evidence(
            FIXTURE.case.incident, start=FIXTURE.start, end=FIXTURE.end))
        while model.calls == 0:
            await asyncio.sleep(.001)
        task.cancel()
        with pytest.raises(DiagnosisCancelled) as caught:
            await task
        return caught.value.state

    state = asyncio.run(cancel())
    assert state.status is RunStatus.CANCELLED
    assert state.verification.error_code == 'cancelled'
    assert state.model_calls == 3 and model.calls == 1
    assert trace_issues(state) == []


@pytest.mark.parametrize("entrypoint", ["api", "worker", "evaluation"])
@pytest.mark.parametrize("verdict", ["supported", "insufficient"])
def test_online_gate_through_real_http_entrypoints(
        tmp_path, monkeypatch, entrypoint, verdict):
    from uuid import UUID
    from test_mcp_http import (
        http_server, ScriptedModel, START, END, INCIDENT, harness,
    )
    from tracemind.api.app import create_app, get_request_settings
    from tracemind.agent.storage.run_repository import RunRepository
    from tracemind.agent.worker import RunWorker
    from tracemind.evaluation.runner import execute_agent

    body = review(verdict)
    body["claims"][0]["text"] = "Payment failed"
    if verdict == "supported":
        body["claims"][0]["supporting_ids"] = [str(UUID(int=1))]
    model = ReviewModel(body=body)
    monkeypatch.setattr(
        "tracemind.agent.verification.create_chat_model",
        lambda settings: model)
    monkeypatch.setattr(
        "tracemind.agent.factory.create_chat_model",
        lambda settings: ScriptedModel())

    async def exercise():
        expected = "complete" if verdict == "supported" else "inconclusive"
        async with http_server() as url:
            settings = Settings(_env_file=None, verifier_enabled=True,
                                telemetry_mcp_url=url,
                                agent_run_database=tmp_path / "runs.db",
                                agent_checkpoint_database=(
                                    tmp_path / "checkpoints.db"),
                                agent_trace_directory=tmp_path / "traces")
            if entrypoint == "api":
                app = create_app()
                app.dependency_overrides[get_request_settings] = (
                    lambda: settings)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://test",
                ) as client:
                    response = await client.post("/api/v1/diagnoses", json={
                        "incident": INCIDENT.model_dump(mode="json"),
                        "investigation_start": START.isoformat(),
                        "investigation_end": END.isoformat()})
                assert response.status_code == 200
                assert response.json()["status"] == expected
            elif entrypoint == "worker":
                repository = RunRepository(settings.agent_run_database)
                state = harness().state.model_copy(update={
                    "status": RunStatus.QUEUED,
                    "verification_policy": settings.verification_policy})
                repository.create(state, request_hash="verifier-http")
                worker = RunWorker(settings, repository=repository)
                assert await worker.run_once()
                stored = repository.get(state.id)
                assert stored.result.diagnosis.status.value == expected
                assert stored.state.verification.status == "completed"
                assert stored.state.model_calls == 3
                assert trace_issues(stored.state) == []
            else:
                result = await execute_agent(
                    INCIDENT, settings, start=START, end=END)
                assert result.error_type is None
                assert result.diagnosis.status.value == expected
                assert result.run_state.verification.status == "completed"
                assert trace_issues(result.run_state) == []
        assert model.calls == 1

    asyncio.run(exercise())


def test_experiment_cap_denies_verifier_before_provider_execution(monkeypatch):
    from tracemind.evaluation.budget import ModelCallBudget
    budget = ModelCallBudget(2)
    model = ReviewModel(body=review())
    service, investigator = make_service(monkeypatch, model)
    service._runner = LangChainAgentRunner(investigator, callbacks=[budget])
    service._verifier.callbacks = [budget]
    with pytest.raises(DiagnosisError) as caught:
        diagnose(service)
    assert budget.calls == 2 and model.calls == 0
    assert caught.value.state.stop_reason is StopReason.VERIFICATION_FAILED
    assert caught.value.state.verification.error_code == "model_unavailable"


def test_review_rejects_a_known_but_out_of_window_observation():
    from datetime import timedelta
    from tracemind.agent.verification import validate_review
    from tracemind.models.verification import DiagnosisReview, ReviewSubject
    subject = ReviewSubject.model_validate(json.loads(
        Path("evals/langfuse/verifier-calibration.json").read_text()
    )[0]["input"])
    claim = DiagnosisReview.model_validate({
        **review(), "claims": [{
            **review()["claims"][0],
            "text": subject.diagnosis.root_cause,
            "supporting_ids": [str(subject.evidence[0].id)],
        }],
    })
    subject.evidence[0].observed_at = subject.start - timedelta(seconds=1)
    with pytest.raises(ValueError, match="out-of-window"):
        validate_review(claim, subject)


def test_saved_and_live_review_preserve_knowledge_and_counterevidence():
    from tracemind.agent.verification import subject_for
    from tracemind.evaluation.models import ExperimentOutput
    from tracemind.models.run import RunState
    from tracemind.models.verification import ReviewSubject
    item = next(item for item in json.loads(
        Path("evals/langfuse/verifier-calibration.json").read_text())
        if item["input"]["knowledge"])
    subject = ReviewSubject.model_validate(item["input"])
    state = RunState(incident=subject.incident,
                     limits=Settings(_env_file=None).run_limits,
                     allowed_tools=(),
                     investigation_start=subject.start,
                     investigation_end=subject.end,
                     knowledge=subject.knowledge)
    saved = ExperimentOutput.from_saved({
        "diagnosis": subject.diagnosis.model_dump(mode="json"),
        "evidence": [e.model_dump(mode="json") for e in subject.evidence],
        "run_state": state.model_dump(mode="json"),
    })
    canonical = [item.model_dump(mode="json") for item in state.knowledge]
    assert saved.knowledge == canonical
    assert saved.knowledge[0]["hit"]["chunk"]["text"] == (
        subject.knowledge[0]["hit"]["chunk"]["text"])
    reviewed = subject_for(state, subject.diagnosis, subject.evidence)
    assert reviewed.knowledge == canonical
    assert reviewed.evidence == subject.evidence


@pytest.mark.parametrize("invented", [
    "Payment's database pool was exhausted",
    "Inspection will restore all requests",
    "No payment failure occurred",
    "   ",
])
def test_reviewer_cannot_add_assertions_absent_from_the_answer(
        monkeypatch, invented):
    body = review()
    body["claims"][0]["text"] = invented
    model = ReviewModel(body=body)
    service, _ = make_service(monkeypatch, model)
    with pytest.raises(DiagnosisError) as caught:
        diagnose(service)
    assert caught.value.state.verification.error_code == "invalid_review"
    assert caught.value.state.diagnosis is None
    assert model.calls == 1


def test_explicit_abstention_with_gaps_is_not_an_asserted_mechanism():
    from tracemind.agent.verification import validate_review
    from tracemind.models.verification import DiagnosisReview, ReviewSubject
    item = next(item for item in json.loads(
        Path("evals/langfuse/verifier-calibration.json").read_text())
        if item["metadata"]["case_id"] == "appropriate_abstention")
    subject = ReviewSubject.model_validate(item["input"])
    body = review()
    body["claims"] = [{
        "text": subject.diagnosis.root_cause,
        "kind": "uncertainty", "verdict": "supported",
        "missing_observations": ["Underlying failure mechanism"],
        "rationale": "The original answer explicitly abstains",
    }]
    result = DiagnosisReview.model_validate(body)
    validate_review(result, subject)
    assert result.accepted


def test_deepseek_review_sets_json_mode_on_the_actual_provider_payload():
    from tracemind.agent.runtime.llm import create_chat_model
    from tracemind.agent.verification import review_options
    settings = Settings(_env_file=None, llm_provider="deepseek",
                        deepseek_api_key="synthetic-placeholder")
    model = create_chat_model(settings)
    payload = model._get_request_payload(
        [{"role": "user", "content": "Return JSON"}],
        **review_options(model, "deepseek", 4096))
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_tokens"] == 4096
    assert payload["extra_body"]["thinking"] == {"type": "disabled"}
