"""Exercise the real SDK with in-memory export and no model/network calls."""

import asyncio
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from langfuse import Evaluation, Langfuse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from tracemind.config import Settings
from tracemind.evaluation.budget import ModelCallBudget
from tracemind.evaluation.evaluators import (
    acceptance_evaluator,
    diagnosis_scores,
    retrieval_scores,
)
from tracemind.evaluation.experiments import (
    accepted,
    make_task,
    run_experiment,
    verify_publication,
)
from tracemind.evaluation.judge import diagnosis_judge
from tracemind.evaluation.models import ExperimentOutput
from tracemind.evaluation.runner import execute_service, execute_agent
from tracemind.agent.errors import ModelConfigurationError
from tracemind.agent.service import DiagnosisService
from tracemind.models import Diagnosis, Evidence, Incident
from tracemind.models.run import RunStatus
from tracemind.observability.langfuse import client_for
from replay_support import ReplayFixture, ReplayTelemetry
from tool_fixtures import FixtureToolClient


@pytest.fixture
def sdk():
    exporter = InMemorySpanExporter()
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"successes": [], "errors": []}
        )
    )
    client = Langfuse(
        public_key="pk-test-" + uuid4().hex,
        secret_key="sk-test",
        span_exporter=exporter,
        httpx_client=httpx.Client(transport=transport),
    )
    yield client, exporter
    client.shutdown()


def saved():
    start = datetime(2024, 9, 1, tzinfo=UTC)
    evidence = Evidence(
        source="trace",
        service="payment",
        summary="Charge failed",
        observed_at=start,
    )
    diagnosis = Diagnosis(
        incident_id=uuid4(),
        status="complete",
        confidence=0.9,
        root_cause="payment Charge failed",
        evidence_ids=[evidence.id],
    )
    return ExperimentOutput(
        diagnosis=diagnosis,
        evidence=[evidence],
        investigation_start=start,
        investigation_end=start + timedelta(hours=1),
    )


def scores(items):
    return {item.name: item.value for item in items}


def test_real_sdk_missing_items_or_evaluators_cannot_pass(sdk):
    client, _ = sdk

    def fail(*, item, **kwargs):
        raise ValueError("synthetic task failure")

    result = client.run_experiment(
        name="task-failure",
        data=[{"input": "x"}],
        task=fail,
        run_evaluators=[acceptance_evaluator(1, {"ok": 1})],
    )
    assert not accepted(result)
    assert scores(result.run_evaluations)["evaluation_coverage"] == 0

    def broken(*, output, **kwargs):
        raise ValueError("synthetic evaluator failure")

    result = client.run_experiment(
        name="eval-failure",
        data=[{"input": "x"}],
        task=lambda **kwargs: "answer",
        evaluators=[broken],
        run_evaluators=[acceptance_evaluator(1, {"ok": 1})],
    )
    assert len(result.item_results) == 1
    assert not accepted(result)


def test_saved_experiment_uses_sdk_scores_and_omits_old_scores(sdk):
    client, exporter = sdk
    observed = saved().model_dump(mode="json")
    observed["result"] = {"passed": True, "secret_label": "DO_NOT_REVIEW"}
    result = run_experiment(
        Settings(_env_file=None),
        client,
        [
            {
                "input": observed,
                "expected_output": {"expected_status": "complete"},
            }
        ],
        kind="saved",
        name="saved-contract",
        budget=ModelCallBudget(1),
    )
    assert accepted(result)
    assert "result" not in result.item_results[0].output
    assert (
        scores(result.item_results[0].evaluations)["citation_integrity"]
        is True
    )
    assert scores(result.run_evaluations)["model_calls_admitted"] == 0
    assert exporter.get_finished_spans()


def test_reference_membership_does_not_claim_causal_correctness():
    observed = saved()
    observed.diagnosis.root_cause = (
        "Not payment; an unobserved mechanism caused it"
    )
    result = scores(diagnosis_scores(output=observed))
    assert result["citation_integrity"]
    assert "root_cause_correctness" not in result
    observed.diagnosis.evidence_ids = [uuid4()]
    assert not scores(diagnosis_scores(output=observed))["citation_integrity"]
    observed.diagnosis.evidence_ids = [observed.evidence[0].id]
    observed.evidence[0].observed_at = observed.investigation_end + timedelta(
        seconds=1
    )
    assert not scores(diagnosis_scores(output=observed))["citation_integrity"]


def test_retrieval_metrics_use_stable_document_ids_and_compare_stages():
    output = ExperimentOutput(
        kind="retrieval",
        retrieval={
            "k": 2,
            "stages": {
                "fused": [
                    {"evaluation_id": "noise"},
                    {"evaluation_id": "useful"},
                ],
                "final": [
                    {"evaluation_id": "useful"},
                    {"evaluation_id": "useful"},
                ],
            },
        },
    )
    result = scores(
        retrieval_scores(
            output=output, expected_output={"relevance": {"useful": 2}}
        )
    )
    assert result["final.recall@2"] == 1
    assert result["final.precision@2"] == 0.5
    assert result["final.mrr@2"] == 1
    assert result["fused.mrr@2"] == 0.5
    assert result["final.ndcg@2"] > result["fused.ndcg@2"]
    assert "final.recall@2" not in scores(retrieval_scores(output=output))


class FakeModel(BaseChatModel):
    response: str = "{}"
    entered: list = []

    @property
    def _llm_type(self):
        return "fixture"

    def _generate(self, messages, **kwargs):
        raise AssertionError("async only")

    async def _agenerate(self, messages, **kwargs):
        self.entered.append(messages)
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=self.response))
            ]
        )


def valid_review(observed, **updates):
    result = {
        "claims": [
            {
                "text": "Charge failed",
                "verdict": "supported",
                "supporting_ids": [str(observed.evidence[0].id)],
                "rationale": "error span",
            }
        ],
        "uncertainty_appropriate": True,
        "root_cause_correctness": None,
        "process_quality": None,
        "rationale": "Observed error supports the claim",
    }
    result.update(updates)
    return result


def test_judge_validates_references_and_does_not_retry(monkeypatch, sdk):
    observed = saved()
    body = valid_review(observed)
    body["claims"][0]["supporting_ids"] = [str(uuid4())]
    model = FakeModel(response=json.dumps(body))
    monkeypatch.setattr(
        "tracemind.evaluation.judge.create_chat_model", lambda settings: model
    )
    budget = ModelCallBudget(2)
    evaluate = diagnosis_judge(Settings(_env_file=None), sdk[0], budget)
    result = asyncio.run(evaluate(output=observed.model_dump(mode="json")))
    assert scores(result) == {"judge_completed": False}
    assert budget.calls == len(model.entered) == 1


def test_judge_keeps_unknown_correctness_unknown_and_rejects_oversized_input(
    monkeypatch, sdk
):
    observed = saved()
    model = FakeModel(response=json.dumps(valid_review(observed)))
    monkeypatch.setattr(
        "tracemind.evaluation.judge.create_chat_model", lambda settings: model
    )
    budget = ModelCallBudget(2)
    evaluate = diagnosis_judge(Settings(_env_file=None), sdk[0], budget)
    result = scores(asyncio.run(evaluate(output=observed)))
    assert result["judge_completed"] and result["evidence_support"] == 1
    assert "root_cause_correctness" not in result
    evaluate = diagnosis_judge(
        Settings(_env_file=None, eval_judge_input_bytes=1), sdk[0], budget
    )
    assert not scores(asyncio.run(evaluate(output=observed)))[
        "judge_completed"
    ]
    assert budget.calls == 1


def test_budget_is_shared_and_denies_before_provider_execution():
    budget = ModelCallBudget(2)
    primary, fallback, judge = FakeModel(), FakeModel(), FakeModel()

    async def invoke():
        for model in (primary, fallback):
            await model.ainvoke("test", config={"callbacks": [budget]})
        with pytest.raises(RuntimeError, match="budget exhausted"):
            await judge.ainvoke("test", config={"callbacks": [budget]})

    asyncio.run(invoke())
    assert budget.calls == 2
    assert len(primary.entered) == len(fallback.entered) == 1
    assert not judge.entered


def test_task_never_passes_expected_output_to_investigator(monkeypatch):
    captured = []

    async def execute(incident, settings, **kwargs):
        captured.append((incident, kwargs))
        return saved()

    monkeypatch.setattr(
        "tracemind.evaluation.experiments.execute_agent", execute
    )
    task = make_task(
        "agent", Settings(_env_file=None), None, ModelCallBudget(1)
    )
    item = {
        "input": {
            "incident": Incident(
                title="incident", description="visible"
            ).model_dump(mode="json"),
            "start": "2024-09-01T00:00:00Z",
            "end": "2024-09-01T01:00:00Z",
        },
        "expected_output": {"reference_answer": "HELD_OUT"},
        "metadata": {"label": "HELD_OUT"},
    }
    asyncio.run(task(item=item))
    assert "HELD_OUT" not in str(captured)


def test_configuration_failure_and_cancellation_retain_run_output(
    tmp_path, monkeypatch
):
    fixture = ReplayFixture.load(Path("evals/replays/payment_failure.json"))

    def unavailable(settings):
        raise ModelConfigurationError("missing synthetic key")

    monkeypatch.setattr(
        "tracemind.agent.factory.create_chat_model", unavailable
    )
    output = asyncio.run(
        execute_agent(
            fixture.case.incident,
            Settings(_env_file=None),
            start=fixture.start,
            end=fixture.end,
        )
    )
    assert (
        output.error_type == "ModelConfigurationError"
        and output.run_state is None
    )
    entered = asyncio.Event()

    class WaitingRunner:
        async def run(self, *args, **kwargs):
            entered.set()
            await asyncio.Future()

    telemetry = ReplayTelemetry([])
    service = DiagnosisService(
        WaitingRunner(),
        tool_client=FixtureToolClient(telemetry, telemetry, telemetry),
    )

    async def cancel():
        task = asyncio.create_task(
            execute_service(
                fixture.case.incident,
                service,
                start=fixture.start,
                end=fixture.end,
                model_provider="fixture",
                model_name="waiting",
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        return await task

    output = asyncio.run(cancel())
    assert output.error_type == "CancelledError"
    assert output.run_state.status is RunStatus.CANCELLED
    assert not scores(diagnosis_scores(output=output))["execution_success"]


def test_missing_publication_fails_and_disabled_runtime_has_no_client():
    assert client_for(Settings(_env_file=None)) is None
    with pytest.raises(ValueError, match="must be true"):
        client_for(Settings(_env_file=None), required=True)
    item = SimpleNamespace(
        trace_id="trace", evaluations=[Evaluation(name="ok", value=True)]
    )
    result = SimpleNamespace(
        item_results=[item], dataset_run_id=None, run_evaluations=[]
    )
    client = SimpleNamespace(
        api=SimpleNamespace(
            scores_v3=SimpleNamespace(
                get_many_v3=lambda **kwargs: SimpleNamespace(data=[])
            )
        )
    )
    assert not verify_publication(client, result, timeout_seconds=0)


def test_qualified_answer_can_support_uncertainty_without_inventing_a_cause(
    monkeypatch, sdk
):
    observed = saved()
    observed.diagnosis.root_cause = (
        "Charge failed; the underlying mechanism is unknown."
    )
    body = valid_review(observed)
    body["claims"].append(
        {
            "text": "the underlying mechanism is unknown",
            "kind": "uncertainty",
            "verdict": "supported",
            "supporting_ids": [],
            "missing_observations": ["Database observations"],
            "rationale": "Answer explicitly qualifies the mechanism",
        }
    )
    model = FakeModel(response=json.dumps(body))
    monkeypatch.setattr(
        "tracemind.evaluation.judge.create_chat_model", lambda settings: model
    )
    result = scores(
        asyncio.run(
            diagnosis_judge(
                Settings(_env_file=None), sdk[0], ModelCallBudget(1)
            )(output=observed)
        )
    )
    assert result["judge_completed"] and result["evidence_support"] == 1
    assert result["false_completion"] is False


def test_mixed_label_dataset_still_enforces_each_labeled_status(sdk):
    observed = saved().model_dump(mode="json")
    result = run_experiment(
        Settings(_env_file=None),
        sdk[0],
        [
            {"input": observed},
            {
                "input": observed,
                "expected_output": {"expected_status": "inconclusive"},
            },
        ],
        kind="saved",
        name="mixed-labels",
        budget=ModelCallBudget(1),
    )
    assert not accepted(result)


def test_publication_requires_matching_scores_and_an_observation():
    item = SimpleNamespace(
        trace_id="trace", evaluations=[Evaluation(name="ok", value=True)]
    )
    result = SimpleNamespace(
        item_results=[item], dataset_run_id=None, run_evaluations=[]
    )
    rows = []
    score = SimpleNamespace(name="ok", value=False)
    client = SimpleNamespace(
        api=SimpleNamespace(
            scores_v3=SimpleNamespace(
                get_many_v3=lambda **kwargs: SimpleNamespace(data=[score])
            ),
            observations=SimpleNamespace(
                get_many=lambda **kwargs: SimpleNamespace(data=rows)
            ),
        )
    )
    assert not verify_publication(client, result, timeout_seconds=0)
    score.value = True
    assert not verify_publication(client, result, timeout_seconds=0)
    rows.append(SimpleNamespace(id="observation"))
    assert verify_publication(client, result, timeout_seconds=0)


def test_langfuse_url_is_normalized_before_constructing_exporter(monkeypatch):
    captured = []
    monkeypatch.setattr(
        "tracemind.observability.langfuse._client",
        lambda *args: captured.append(args) or "client",
    )
    assert (
        client_for(
            Settings(
                _env_file=None,
                langfuse_enabled=True,
                langfuse_public_key="pk-synthetic",
                langfuse_secret_key="sk-synthetic",
                langfuse_base_url="http://127.0.0.1:3000/",
            )
        )
        == "client"
    )
    assert captured[0][2] == "http://127.0.0.1:3000"


@pytest.mark.parametrize("grade", [True, float("nan"), float("inf"), -1, 4])
def test_retrieval_rejects_invalid_relevance_grades(grade):
    with pytest.raises(ValueError, match="grades"):
        retrieval_scores(
            output=ExperimentOutput(kind="retrieval", retrieval={"k": 5}),
            expected_output={"relevance": {"synthetic": grade}},
        )


def test_judge_timeout_does_not_retry_or_pass(monkeypatch, sdk):
    class SlowModel(FakeModel):
        async def _agenerate(self, messages, **kwargs):
            self.entered.append(messages)
            await asyncio.Future()

    model = SlowModel()
    monkeypatch.setattr(
        "tracemind.evaluation.judge.create_chat_model", lambda settings: model
    )
    budget = ModelCallBudget(2)
    evaluate = diagnosis_judge(
        Settings(_env_file=None, eval_judge_timeout_seconds=0.01),
        sdk[0],
        budget,
    )
    assert scores(asyncio.run(evaluate(output=saved()))) == {
        "judge_completed": False
    }
    assert budget.calls == len(model.entered) == 1


def test_export_failure_does_not_change_runtime_diagnosis(tmp_path):
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
    from replay_support import ScriptedReplayModel
    from tracemind.agent.runtime.runner import LangChainAgentRunner

    class OfflineExporter(SpanExporter):
        attempts = 0

        def export(self, spans):
            self.attempts += 1
            return SpanExportResult.FAILURE

    exporter = OfflineExporter()
    client = Langfuse(
        public_key="pk-offline-" + uuid4().hex,
        secret_key="sk-synthetic",
        span_exporter=exporter,
    )
    fixture = ReplayFixture.load(Path("evals/replays/payment_failure.json"))
    service = DiagnosisService(
        LangChainAgentRunner(ScriptedReplayModel(script=fixture.script)),
        tool_client=ReplayTelemetry(fixture.responses),
        event_directory=tmp_path,
        langfuse_client=client,
    )
    try:
        output = asyncio.run(
            execute_service(
                fixture.case.incident,
                service,
                start=fixture.start,
                end=fixture.end,
                model_provider="synthetic",
                model_name="scripted",
            )
        )
        client.flush()
        assert output.diagnosis.status.value == "complete"
        assert (
            scores(diagnosis_scores(output=output))["trace_integrity"] is True
        )
        assert exporter.attempts > 0
    finally:
        client.shutdown()


def test_process_claim_uses_execution_events_without_claiming_incident_support(
    monkeypatch, sdk, tmp_path
):
    from replay_support import run_replay_evaluation

    observed = asyncio.run(
        run_replay_evaluation(
            ReplayFixture.load(Path("evals/replays/payment_failure.json")),
            Settings(_env_file=None, agent_trace_directory=tmp_path),
        )
    )
    sequence = next(
        event.sequence
        for event in observed.run_state.events
        if event.kind == "tool.completed"
    )
    observed.diagnosis.root_cause += (
        "; The trace query returned an observation")
    body = valid_review(observed)
    body["claims"].append(
        {
            "text": "The trace query returned an observation",
            "kind": "process",
            "verdict": "supported",
            "supporting_event_sequences": [sequence],
            "rationale": "Tool completion records this operation",
        }
    )
    body["process_quality"] = 0.9
    model = FakeModel(response=json.dumps(body))
    monkeypatch.setattr(
        "tracemind.evaluation.judge.create_chat_model", lambda settings: model
    )
    evaluate = diagnosis_judge(
        Settings(_env_file=None), sdk[0], ModelCallBudget(2)
    )
    assert scores(asyncio.run(evaluate(output=observed)))["judge_completed"]
    body["claims"][-1]["supporting_event_sequences"] = [999999]
    model.response = json.dumps(body)
    assert not scores(asyncio.run(evaluate(output=observed)))[
        "judge_completed"
    ]


def test_first_runtime_callback_initializes_its_project_before_generation(
    monkeypatch,
):
    from tracemind.observability.langfuse import callbacks_for

    exporter = InMemorySpanExporter()
    clients = []

    def initialize(public_key, secret_key, base_url, environment):
        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
            span_exporter=exporter,
        )
        clients.append(client)
        return client

    monkeypatch.setattr("tracemind.observability.langfuse._client", initialize)
    settings = Settings(
        _env_file=None,
        langfuse_enabled=True,
        langfuse_public_key="pk-fresh-" + uuid4().hex,
        langfuse_secret_key="sk-synthetic",
    )
    try:
        callbacks = callbacks_for(settings)
        asyncio.run(
            FakeModel().ainvoke(
                "fabricated example", config={"callbacks": callbacks}
            )
        )
        assert clients
        clients[0].flush()
        assert any(
            span.attributes.get("langfuse.observation.type") == "generation"
            for span in exporter.get_finished_spans()
        )
    finally:
        for client in clients:
            client.shutdown()


def test_vector_only_retrieval_does_not_require_a_provider_key(
    monkeypatch, sdk
):
    from tracemind.rag.config import RagSettings

    class VectorOnlyRuntime:
        def __init__(self, *args, **kwargs):
            self.last_retrieval = {
                "k": 5,
                "stages": {"final": []},
                "model_calls": 0,
            }

        async def initialize(self):
            pass

        async def search(self, query, **kwargs):
            return []

        async def close(self):
            pass

    def forbidden(settings):
        raise AssertionError("vector-only retrieval must not construct an LLM")

    monkeypatch.setattr("tracemind.rag.engine.RagRuntime", VectorOnlyRuntime)
    monkeypatch.setattr(
        "tracemind.agent.runtime.llm.create_chat_model", forbidden
    )
    budget = ModelCallBudget(1)
    task = make_task(
        "retrieval",
        Settings(_env_file=None),
        sdk[0],
        budget,
        rag_settings=RagSettings(_env_file=None, mode="naive"),
    )
    output = asyncio.run(task(item={"input": {"query": "manufactured query"}}))
    assert output["retrieval"]["model_calls"] == budget.calls == 0
    assert scores(retrieval_scores(output=output))["execution_success"]


def test_mixed_saved_dataset_cannot_disable_an_existing_items_trace_gate(
        sdk, tmp_path):
    from replay_support import run_replay_evaluation
    observed = asyncio.run(run_replay_evaluation(
        ReplayFixture.load(Path("evals/replays/payment_failure.json")),
        Settings(_env_file=None, agent_trace_directory=tmp_path),
    )).model_dump(mode="json")
    observed["run_state"]["events"] = []
    result = run_experiment(
        Settings(_env_file=None), sdk[0],
        [{"input": observed}, {"input": dict(observed, run_state=None)}],
        kind="saved", name="mixed-trace-regression", budget=ModelCallBudget(1),
    )
    assert scores(result.item_results[0].evaluations)[
        "trace_integrity"] is False
    assert not accepted(result)


def test_verification_accepts_correct_rejection_with_held_out_labels(
        monkeypatch, sdk):
    observed = saved()
    incident = Incident(
        id=observed.diagnosis.incident_id,
        title="synthetic",
        description="synthetic")
    body = valid_review(observed)
    body["claims"][0].update(verdict="insufficient", supporting_ids=[],
                             missing_observations=["A linked operation"])
    model = FakeModel(response=json.dumps(body))
    monkeypatch.setattr(
        "tracemind.agent.verification.create_chat_model",
        lambda settings: model)
    supplied = {"incident": incident.model_dump(mode="json"),
                "diagnosis": observed.diagnosis.model_dump(mode="json"),
                "evidence": [e.model_dump(mode="json")
                             for e in observed.evidence],
                "start": observed.investigation_start.isoformat(),
                "end": observed.investigation_end.isoformat()}
    result = run_experiment(
        Settings(_env_file=None), sdk[0],
        [{"input": supplied, "expected_output": {"verdict": "insufficient"},
          "metadata": {"private_label": "HELD_OUT_LABEL"}}],
        kind="verification", name="correct-rejection",
        budget=ModelCallBudget(1),
    )
    assert accepted(result)
    item_scores = scores(result.item_results[0].evaluations)
    assert item_scores["expected_decision"] and not item_scores["would_accept"]
    assert item_scores["false_acceptance"] is False
    assert "HELD_OUT_LABEL" not in str(model.entered)
    subject = json.loads(model.entered[0][-1].text)
    assert subject["reference"] is None
    assert scores(result.run_evaluations)["model_calls_admitted"] == 1
