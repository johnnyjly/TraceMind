"""Regressions for role prompts, reviewer context and model profiles."""

import asyncio
from copy import deepcopy
import json

import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage

from tracemind.agent.runtime.audit import trace_issues
from tracemind.agent.runtime.middleware import HarnessMiddleware
from tracemind.agent.runtime.rag_model_factory import create_rag_completion
from tracemind.agent.verification import create_review_model, review_options
from tracemind.config import Settings
from tracemind.evaluation.budget import ModelCallBudget
from tracemind.evaluation.experiments import accepted, run_experiment
from tracemind.evaluation.models import ExperimentOutput
from tracemind.models.run import StopReason
from tracemind.models.verification import VerificationPolicy
from tracemind.rag.config import RagSettings
from test_langfuse import FakeModel, saved, sdk, valid_review  # noqa: F401
from test_recovery import harness


@pytest.mark.parametrize("purpose", ["retrieval", "ingestion"])
def test_retrieval_profile_isolates_thinking_json_and_output(
    monkeypatch,
    purpose,
):
    created, bindings, requests = [], [], []

    class Model:
        def bind(self, **kwargs):
            bindings.append(kwargs)
            return self

        async def ainvoke(self, messages, **kwargs):
            requests.append(kwargs)
            return AIMessage(content="{}")

    def factory(settings):
        created.append(settings)
        return Model()

    monkeypatch.setattr(
        "tracemind.agent.runtime.llm.create_chat_model", factory
    )
    settings = Settings(_env_file=None, llm_provider="deepseek")
    complete = create_rag_completion(
        settings, RagSettings(_env_file=None), purpose=purpose
    )
    assert not created

    async def exercise():
        for keyword in (False, True):
            await complete([], max_tokens=2048, keyword_extraction=keyword)

    asyncio.run(exercise())
    retrieval = purpose == "retrieval"
    assert len(created) == 1
    assert created[0].deepseek_thinking is (not retrieval)
    assert settings.deepseek_thinking is True
    assert len(bindings) == (2 if retrieval else 1)
    assert all(
        b == {"response_format": {"type": "json_object"}} for b in bindings
    )
    assert requests == [{"max_tokens": 512 if retrieval else 2048}] * 2


@pytest.mark.parametrize(
    "output", ["CoordinatorDecision", "InvestigationReport"]
)
def test_specialist_closure_uses_assigned_schema(output):
    run = harness()
    run.begin_finalization(StopReason.TOOL_BUDGET)
    middleware = HarnessMiddleware(run, output_name=output)
    request = ModelRequest(
        model=FakeModel(), messages=[HumanMessage("Task")], tools=[], state={}
    )
    prepared = middleware._prepare_request(request)
    text = prepared.messages[-1].text
    assert text.count("Investigation is closed") == 1
    assert output in text
    assert "diagnosis" not in text.lower()
    assert "inconclusive" not in text
    assert not prepared.tools
    if output == "InvestigationReport":
        assert "missing_observations" in text


@pytest.mark.parametrize("sequence, success", [(17, True), (999, False)])
@pytest.mark.parametrize(
    "event_kind", ["tool.completed", "policy.stopped", "run.finished"]
)
def test_verification_preserves_original_process_events(
    monkeypatch,
    sdk,  # noqa: F811
    sequence,
    success,
    event_kind,  # noqa: F811
):
    observed = saved()
    observed.diagnosis.root_cause = (
        "The trace search returned one observation."
    )
    supplied = {
        "incident": {
            "id": str(observed.diagnosis.incident_id),
            "title": "synthetic",
            "description": "synthetic",
        },
        "diagnosis": observed.diagnosis.model_dump(mode="json"),
        "evidence": [e.model_dump(mode="json") for e in observed.evidence],
        "start": observed.investigation_start.isoformat(),
        "end": observed.investigation_end.isoformat(),
        "execution_events": [
            {
                "sequence": 17,
                "kind": event_kind,
                "timestamp": (observed.investigation_start.isoformat()),
                "data": {
                    "tool": "search_traces",
                    "evidence_ids": [str(observed.evidence[0].id)],
                },
            }
        ],
    }
    original = deepcopy(supplied)
    body = valid_review(observed)
    body["assessments"][0].update(
        kind="process",
        supporting_refs=[],
        supporting_event_sequences=[sequence],
    )
    model = FakeModel(response=json.dumps(body))
    configured = []

    def factory(settings):
        configured.append(settings)
        return model

    monkeypatch.setattr(
        "tracemind.agent.verification.create_review_model", factory
    )
    result = run_experiment(
        Settings(
            _env_file=None,
            deepseek_reasoning_effort="high",
            verifier_reasoning_effort="low",
        ),
        sdk[0],
        [{"input": supplied, "expected_output": {"verdict": "supported"}}],
        kind="verification",
        name="original-process-events",
        budget=ModelCallBudget(2),
    )
    assert accepted(result) is success
    packet = json.loads(model.entered[0][-1].text)
    assert packet["query_events"][0]["sequence"] == 17
    assert packet["query_events"][0]["data"]["evidence_ids"] == ["E1"]
    assert packet["reference"] is None
    assert supplied == original
    output = ExperimentOutput.model_validate(result.item_results[0].output)
    state = output.run_state
    assert trace_issues(state) == []
    assert state.tool_calls == {}
    assert not any(e.kind.startswith("tool.") for e in state.events)
    assert state.model_calls == (1 if success else 2)
    assert output.elapsed_ms == state.elapsed_ms > 0
    assert configured[0].deepseek_reasoning_effort == "low"


def test_review_settings_independent_and_policy_persisted():
    settings = Settings(_env_file=None, deepseek_reasoning_effort="low")
    assert settings.verification_policy.deepseek_reasoning_effort == "high"
    assert settings.eval_judge_reasoning_effort == "high"
    policy = settings.model_copy(
        update={"verifier_reasoning_effort": "low"}
    ).verification_policy
    assert (
        VerificationPolicy.model_validate_json(policy.model_dump_json())
        == policy
    )
    assert VerificationPolicy().deepseek_reasoning_effort == "high"
    model = create_review_model(
        Settings(
            _env_file=None,
            llm_provider="deepseek",
            deepseek_api_key="test",
            deepseek_reasoning_effort="low",
        )
    )
    assert review_options(model, "deepseek", 8192)["reasoning"] == {
        "effort": "low"
    }
    assert model.use_responses_api is True


@pytest.mark.parametrize("reasoning", [None, 0, 7, 21])
def test_reasoning_usage_is_reported_or_unknown(reasoning):
    run = harness()

    async def call():
        return AIMessage(
            content="done",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "output_token_details": (
                    {} if reasoning is None else {"reasoning": reasoning}
                ),
            },
        )

    asyncio.run(run.model_call(call, model_name="test"))
    known = reasoning is not None and reasoning <= 20
    usage = run.state.usage
    assert usage.reported_reasoning_tokens == (reasoning if known else 0)
    assert usage.reported_reasoning_model_calls == int(known)
    assert usage.unreported_reasoning_model_calls == int(not known)
    event = next(e for e in run.state.events if e.kind == "model.completed")
    assert event.data["reasoning_tokens"] == (reasoning if known else None)
    assert usage.reported_total_tokens == 30
