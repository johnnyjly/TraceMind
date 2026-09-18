"""Task packets, provenance and branch-local failures at agent boundaries."""

import asyncio
from hashlib import sha256
import json
from uuid import uuid4

import pytest
from mcp.types import CallToolResult
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import HumanMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from test_dynamic_tools import Client, bind, definition, state
from test_multi_agent import TASK, REPORT, coordinator_packet, service_for, run
from test_recovery import harness
from test_task_budgets import allocate, in_task
from test_verifier import EVIDENCE, ReviewModel, review

from tracemind.agent.errors import DiagnosisValidationError
from tracemind.agent.runtime.collaboration import (
    coordinator_context, validate_report,
)
from tracemind.agent.runtime.harness import TaskToolBudgetExhausted
from tracemind.agent.runtime.middleware import (
    HarnessMiddleware, TaskContextExceeded,
)
from tracemind.agent.runtime.tools import EvidenceCollector
from tracemind.agent.workflow.handoffs import freeze_task_input, role_tools
from tracemind.mcp.results import RESULT_CONTRACT_KEY, TELEMETRY_CONTRACT
from tracemind.models.collaboration import (
    CollaborationState, InvestigationTask, InvestigationReport, TaskRecord,
)
from tracemind.models.run import RunStatus


def test_task_input_is_frozen_and_does_not_copy_parent_or_sibling_context():
    run = harness()
    run.state.collaboration = CollaborationState()
    run.state.previous_messages = ["PRIVATE_PARENT_CONVERSATION"]
    sibling = EVIDENCE.model_copy(update={
        "id": uuid4(), "summary": "PRIVATE_SIBLING_OBSERVATION",
    })
    run.state.reference_map = {"E1": str(EVIDENCE.id), "E2": str(sibling.id)}
    task = InvestigationTask(
        **TASK, context_refs=["E1"],
        acceptance_criteria=["Find operation", "Find control"],
    )
    packet = freeze_task_input(run.state, "a", task, [EVIDENCE, sibling],
                               ["read_evidence"])
    original = packet.prompt
    assert "PRIVATE_PARENT_CONVERSATION" not in original
    assert "PRIVATE_SIBLING_OBSERVATION" not in original
    assert packet.context_refs == ("E1",)
    assert json.loads(original)["acceptance_criteria"]["C2"] == "Find control"
    run.state.reference_map["E3"] = str(uuid4())
    assert packet.prompt == original
    assert packet.sha256 == sha256(original.encode()).hexdigest()
    record = TaskRecord(id="a", round=0, task=task, inputs=packet,
                        tool_limit=1, report=InvestigationReport(**REPORT))
    run.state.collaboration.tasks["a"] = record
    packet = coordinator_context(run.state, [EVIDENCE, sibling])
    handed_back = packet["tasks"][0]
    assert handed_back["acceptance_criteria"] == record.inputs.criteria
    assert handed_back["input_sha256"] == record.inputs.sha256


def test_dynamic_telemetry_grant_does_not_admit_generic_remote_tool():
    definitions = [definition("ops.inspect", meta={
        RESULT_CONTRACT_KEY: TELEMETRY_CONTRACT,
    }), definition("deploy.release")]
    tools, _, _ = asyncio.run(bind(
        Client(definitions, CallToolResult(content=[])),
        state(("ops.inspect", "deploy.release", "read_evidence")),
    ))
    assert {t.name for t in role_tools(tools.values(), "telemetry")} == {
        "ops.inspect", "read_evidence",
    }
    assert {t.name for t in role_tools(tools.values(), "knowledge")} == {
        "read_evidence",
    }


def test_local_read_repetition_is_per_task_and_does_not_stop_sibling():
    run = harness(identical_tool_calls=1, tool_calls=20)
    run.state.allowed_tools = ("read_evidence",)
    allocate(run, a=5, b=5)

    async def read():
        return []

    async def one():
        return await run.tool_call("read_evidence", {}, read, remote=False)

    async def scenario():
        await in_task(run, "a", one)
        await in_task(run, "b", one)
        with pytest.raises(TaskToolBudgetExhausted):
            await in_task(run, "a", one)
        run.state.reference_map["E1"] = str(EVIDENCE.id)
        await in_task(run, "a", one)
        assert run.finalization_reason is None

    asyncio.run(scenario())


def test_specialist_context_overflow_leaves_another_thread_admissible():
    run = harness(context_tokens=10000, output_tokens=1000)
    middleware = HarnessMiddleware(run, output_name="InvestigationReport")
    model = FakeListChatModel(responses=["unused"])
    oversized = ModelRequest(
        model=model, messages=[HumanMessage(content="x" * 20000)],
    )
    with pytest.raises(TaskContextExceeded):
        middleware._prepare_request(oversized)
    assert run.finalization_reason is None
    run.check_active()
    prepared = middleware._prepare_request(
        ModelRequest(model=model, messages=[HumanMessage(content="Other")])
    )
    assert prepared.model_settings["max_tokens"] == 1000


def test_empty_or_wrong_criterion_report_cannot_be_accepted():
    run = harness()
    run.state.collaboration = CollaborationState()
    task = InvestigationTask(**TASK)
    inputs = freeze_task_input(run.state, "a", task, [], [])
    record = TaskRecord(id="a", round=0, task=task, inputs=inputs,
                        tool_limit=0)
    for body in [
        {"findings": [], "missing_observations": [],
         "answered_criteria": ["C1"]},
        {"findings": [], "missing_observations": [],
         "answered_criteria": ["C2"]},
    ]:
        with pytest.raises(DiagnosisValidationError):
            validate_report(InvestigationReport(**body), run.state, run,
                            record)
    validate_report(InvestigationReport(
        findings=[], missing_observations=["No current observations"],
        outcome="blocked", reason_code="missing_data",
    ), run.state, run, record)


def test_evidence_identity_cannot_be_rewritten_after_publication():
    collector = EvidenceCollector(initial=[EVIDENCE])
    with pytest.raises(ValueError, match="different content"):
        collector.add([EVIDENCE.model_copy(update={"summary": "rewritten"})])
    assert collector.evidence[0] == EVIDENCE


def unmet_completion():
    return [{"criterion_id": key, "met": key != "mechanism",
             "item_ids": ["F1"],
             "missing_requirements": (["UNREVIEWED_ASSERTION"]
                                      if key == "mechanism" else []),
             "rationale": "Explicit scripted completion judgement"}
            for key in ("owner", "operation", "mechanism", "symptom_link")]


@pytest.mark.parametrize("status", ["complete", "inconclusive"])
def test_supported_claims_do_not_imply_completed_task(monkeypatch, status):
    from test_multi_agent import response, submit, DRAFT

    def respond(messages, calls):
        packet = coordinator_packet(messages)
        if packet and packet["originals"]:
            return submit({**DRAFT, "status": status})
        return response(messages, calls)

    reviewer = ReviewModel(body=review(completion=unmet_completion()))
    service, _, _, _ = service_for(
        monkeypatch, reviewer=reviewer, respond=respond,
    )
    result = run(service)
    assert result.state.status is RunStatus.INCONCLUSIVE
    assert result.state.verification.review.accepted
    assert not result.state.verification.review.task_completed
    assert "completion-mechanism" in result.diagnosis.root_cause
    assert "UNREVIEWED_ASSERTION" not in result.diagnosis.root_cause
    assert "Payment Charge failed" in result.diagnosis.root_cause
