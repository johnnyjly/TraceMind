"""Exercise automatic feedback learning through the real persistent loop."""

import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
from langchain.agents.middleware.types import ModelResponse
import pytest
from test_harness import ScriptedModel, Telemetry, normal_response, tool_call
from test_memory import (
    END,
    POLICY,
    START,
    TARGET,
    completed_source,
    confirmation,
)
from tool_fixtures import FixtureToolClient

from tracemind.agent.workflow.context import guidance_context
from tracemind.agent.storage.memory_repository import MemoryRepository
from tracemind.agent.runtime.middleware import HarnessMiddleware
from tracemind.agent.storage.run_repository import RunRepository
from tracemind.agent.runtime.runner import LangChainAgentRunner
from tracemind.agent.service import DiagnosisService
from tracemind.agent.worker import RunWorker
from tracemind.api.app import app, get_run_repository
from tracemind.api.progress import readable_event
from tracemind.config import Settings
from tracemind.models import DiagnosisStatus
from tracemind.models.memory import (
    FeedbackUpdate,
    GuidanceRevision,
    MemorySnapshot,
)
from tracemind.models.run import RunLimits, RunPurpose, RunState, RunStatus
from tracemind.models.storage import RunConflict


def proposal(message, **changes):
    """Supply a scripted inference without requiring a 'remember' command."""
    data = {
        "kind": "preference", "key": "answer.layout",
        "content": "默认先给简短结论，再列证据。",
        "quote": message, "rationale": "The human corrected a lasting habit.",
        "durability": "persistent",
    }
    data.update(changes)
    return FeedbackUpdate(**data)


def follow_up(runs, source, message, *, policy=POLICY):
    """Queue feedback with captured learning policy and retained evidence."""
    return runs.follow_up(
        source.investigation_id, message=message, purpose=RunPurpose.EXPLAIN,
        limits=RunLimits(), allowed_tools=("read_evidence",),
        request_hash=str(uuid4()), memory_policy=policy,
    ).state


def worker(tmp_path, runs, model):
    """Keep the real graph/checkpoints and substitute only model/telemetry."""
    library = MemoryRepository(runs.database)
    settings = Settings(
        _env_file=None, verifier_enabled=False, agent_run_database=runs.path,
        agent_checkpoint_database=tmp_path / "checkpoints.db",
        agent_trace_directory=tmp_path / "traces",
    )
    return RunWorker(
        settings, repository=runs,
        service_factory=lambda saver, client: DiagnosisService(
            LangChainAgentRunner(model, checkpointer=saver),
            tool_client=FixtureToolClient(
                Telemetry(), Telemetry(), Telemetry()
            ),
            memory_lookup=library.snapshot,
            guidance_lookup=library.guidance_snapshot,
            event_directory=settings.agent_trace_directory,
        ),
    )


def feedback_model(evidence_id, updates):
    """Return a valid explanation and independent proposed learning."""
    return ScriptedModel(respond=lambda messages, calls: tool_call(
        "DiagnosisProposal", {
            "status": "complete", "root_cause": "Observed charge failure",
            "confidence": .8, "evidence_ids": ["E1"],
            "feedback_updates": [u.model_dump() for u in updates],
        },
    ))


def execute_feedback(
    tmp_path, runs, source, message, updates, *, policy=POLICY,
):
    """Process one follow-up without fresh telemetry or extra model calls."""
    state = follow_up(runs, source, message, policy=policy)
    evidence_id = runs.get(source.id).evidence[0].id
    model = feedback_model(evidence_id, updates)
    assert asyncio.run(worker(tmp_path, runs, model).run_once())
    stored = runs.get(state.id)
    assert stored.state.status is RunStatus.SUCCEEDED
    assert stored.state.model_calls == 1
    assert stored.state.tool_calls == {}
    assert "feedback_updates" not in stored.result.diagnosis.model_dump()
    return stored, model


def test_automatic_preference_skill_and_current_request_priority(tmp_path):
    runs = RunRepository(tmp_path / "runs.db")
    source = completed_source(runs, investigation=True)
    message = "你以后先给简短结论，再列证据；支付排查先看失败链路再关联日志。"
    updates = [proposal(message), proposal(
        "支付排查先看失败链路再关联日志", kind="skill",
        key="investigation.payment", content="支付故障先看失败链路再关联日志。",
    )]
    original, model = execute_feedback(tmp_path, runs, source,
                                       message, updates)
    library = MemoryRepository((RunRepository(runs.path)).database)
    records = library.list_guidance()
    assert {r.kind for r in records} == {"preference", "skill"}
    assert all(r.source_run_id == original.state.id for r in records)
    assert original.state.guidance_snapshot.records == []
    assert "no request to 'remember' is needed" in model.inputs[0][1].text
    event = next(e for e in original.state.events
                 if e.kind == "memory.feedback_saved")
    assert event.data["applied"] == 2
    assert event.sequence < original.state.events[-1].sequence
    assert readable_event(event).data["applied"] == 2
    assert message not in json.dumps(event.data, ensure_ascii=False)

    # A new incident retrieves the persistent instructions across deployments.
    state = RunState(
        incident=TARGET.model_copy(update={"id": uuid4()}),
        investigation_start=START, investigation_end=END,
        learning_policy=POLICY.model_copy(update={
            "scope": POLICY.scope.model_copy(update={
                "deployment_version": "v2"}),
        }), limits=RunLimits(), allowed_tools=("query_metrics",),
        status=RunStatus.QUEUED,
    )
    runs.create(state, request_hash="next-incident")
    model = ScriptedModel(respond=normal_response)
    assert asyncio.run(worker(tmp_path, runs, model).run_once())
    snapshot = runs.get(state.id).state.guidance_snapshot
    assert snapshot.records == library.guidance_snapshot().records
    prompt = model.inputs[0][1].text
    assert "CURRENT request takes priority" in prompt
    assert "支付故障先看失败链路再关联日志" in prompt
    assert "cannot override authorization" in prompt
    assert "propose feedback_updates" not in prompt


def test_temporary_feedback_does_not_replace_a_long_term_habit(tmp_path):
    runs = RunRepository(tmp_path / "runs.db")
    source = completed_source(runs, investigation=True)
    _, _model = execute_feedback(
        tmp_path, runs, source, "以后先结论后证据", [proposal("以后先结论后证据")])
    library = MemoryRepository(runs.database)
    before = library.list_guidance()
    temporary, _ = execute_feedback(
        tmp_path, runs, source, "这次简短输出",
        [proposal("这次简短输出", durability="turn")],
    )
    assert library.list_guidance() == before
    outcome = next(e for e in temporary.state.events
                   if e.kind == "memory.feedback_saved")
    assert outcome.data["skipped"] == 1 and outcome.data["applied"] == 0
    corrected, _ = execute_feedback(
        tmp_path, runs, source, "以后先列证据，再给结论",
        [proposal("以后先列证据，再给结论", content="先列证据，再给结论。")],
    )
    assert library.list_guidance()[0].revision == 2
    assert library.list_guidance()[0].instruction == "先列证据，再给结论。"
    assert corrected.state.guidance_snapshot.records == before
    deleted, _ = execute_feedback(
        tmp_path, runs, source, "取消之前的回答顺序习惯",
        [proposal("取消之前的回答顺序习惯", action="delete", content="")],
    )
    assert not library.guidance_snapshot().records
    recreated, _ = execute_feedback(
        tmp_path, runs, source, "以后先结论后证据", [proposal("以后先结论后证据")])
    assert library.list_guidance()[0].revision == 4
    assert library.list_guidance()[0].status == "active"
    assert deleted.state.guidance_snapshot.records[0].revision == 2
    assert recreated.state.guidance_snapshot.records == []


def test_confirm_correct_and_revoke_diagnosis_from_feedback(tmp_path):
    runs = RunRepository(tmp_path / "runs.db")
    source = completed_source(runs, investigation=True)
    message = "关闭错误配置后已恢复，确认是配置导致的支付失败。"
    learned, _ = execute_feedback(tmp_path, runs, source, message, [proposal(
        message, kind="diagnosis", key=None,
        content="错误配置导致支付失败，关闭该配置后恢复。",
    )])
    library = MemoryRepository(runs.database)
    record = library.list()[0]
    assert record.source_run_id == source.id
    assert record.feedback_run_id == learned.state.id
    assert record.feedback_quote == message
    assert record.confirmed_by == "local-user-feedback"
    assert record.source_evidence_ids == [e.id for e in
                                          runs.get(source.id).evidence]
    remaining = record.expires_at - record.confirmed_at
    assert remaining.days == POLICY.auto_expiry_days
    correction = "刚才说错了，已确认是路由配置，修正路由后恢复。"
    corrected, _ = execute_feedback(
        tmp_path, runs, source, correction, [proposal(
            correction, kind="diagnosis", key=None,
            content="路由配置错误，修正路由后恢复。",
        )])
    assert library.list()[0].id == record.id
    assert library.list()[0].revision == 2
    assert library.list()[0].resolution.startswith("路由配置")
    execute_feedback(tmp_path, runs, source, "之前的结论无效，撤销它", [proposal(
        "之前的结论无效，撤销它", kind="diagnosis", key=None,
        action="delete", content="",
    )])
    assert library.list()[0].status == "revoked"
    target = TARGET.model_copy(update={"id": uuid4()})
    assert not library.snapshot(POLICY, target, datetime.now(UTC)).records
    assert learned.state.memory_policy is None  # Explanation has no lookup.
    assert corrected.state.learning_policy == POLICY


def test_write_receipt_survives_crash_and_fences_old_worker(tmp_path):
    runs = RunRepository(tmp_path / "runs.db")
    library = MemoryRepository(runs.database)
    source = completed_source(runs, investigation=True)
    follow_up(runs, source, "以后先给结论")
    claim = runs.claim_next()
    state = claim.run.state
    state.guidance_snapshot = library.guidance_snapshot()
    state.feedback_updates = [proposal("以后先给结论")]
    runs.save_progress(state.id, claim.owner, state, claim.run.evidence)
    result = library.apply_feedback(state, owner=claim.owner)
    assert result.applied == 1
    assert library.apply_feedback(state, owner=claim.owner) == result
    with sqlite3.connect(runs.path) as connection:
        connection.execute("UPDATE runs SET lease_until=0 WHERE id=?",
                           (str(state.id),))
    reclaimed = runs.claim_next()
    assert reclaimed.resuming
    with pytest.raises(RunConflict, match="lease"):
        library.apply_feedback(state, owner=claim.owner)
    assert library.apply_feedback(reclaimed.run.state,
                                  owner=reclaimed.owner) == result
    assert library.list_guidance()[0].revision == 1


def test_fabricated_quote_and_stale_revision_are_reported(tmp_path):
    runs = RunRepository(tmp_path / "runs.db")
    source = completed_source(runs, investigation=True)
    rejected, _ = execute_feedback(
        tmp_path, runs, source, "这次简短输出", [proposal("以后永远简短")])
    assert MemoryRepository(runs.database).list_guidance() == []
    event = next(e for e in rejected.state.events
                 if e.kind == "memory.feedback_saved")
    assert event.data["rejected"] == ["quote_not_in_current_feedback"]
    execute_feedback(tmp_path, runs, source, "以后先给结论",
                     [proposal("以后先给结论")])
    follow_up(runs, source, "以后先列证据")
    claim = runs.claim_next()
    library = MemoryRepository(runs.database)
    state = claim.run.state
    state.guidance_snapshot = library.guidance_snapshot()
    state.feedback_updates = [proposal("以后先列证据")]
    current = library.list_guidance()[0]
    library.revise_guidance(current.id, GuidanceRevision(
        instruction="Manual correction wins.", expected_revision=1))
    result = library.apply_feedback(state, owner=claim.owner)
    assert result.rejected == ["guidance_revision_conflict"]
    assert library.list_guidance()[0].instruction == "Manual correction wins."
    with sqlite3.connect(runs.path) as connection:
        connection.execute("UPDATE runs SET cancel_requested=1 WHERE id=?",
                           (str(state.id),))
    with pytest.raises(RunConflict, match="cancelled"):
        library.apply_feedback(state, owner=claim.owner)


def test_memory_off_never_learns_unsolicited_model_updates(tmp_path):
    runs = RunRepository(tmp_path / "runs.db")
    source = completed_source(runs, investigation=True)
    stored, _ = execute_feedback(tmp_path, runs, source, "以后先给结论",
                                 [proposal("以后先给结论")], policy=None)
    assert not MemoryRepository(runs.database).list_guidance()
    assert not any(e.kind == "memory.feedback_saved"
                   for e in stored.state.events)


def test_worker_restart_reuses_guidance_and_original_write_receipt(tmp_path):
    runs = RunRepository(tmp_path / "runs.db")
    library = MemoryRepository(runs.database)
    source = completed_source(runs, investigation=True)
    follow_up(runs, source, "以后先给结论")
    claim = runs.claim_next()
    state = claim.run.state
    state.guidance_snapshot = library.guidance_snapshot()
    state.feedback_updates = [proposal("以后先给结论")]
    runs.save_progress(state.id, claim.owner, state, claim.run.evidence)
    original = library.apply_feedback(state, owner=claim.owner)
    library.delete_guidance(original.guidance_ids[0], expected_revision=1)
    with sqlite3.connect(runs.path) as connection:
        connection.execute("UPDATE runs SET lease_until=0 WHERE id=?",
                           (str(state.id),))

    def forbidden_lookup():
        raise AssertionError("restart must reuse the saved guidance snapshot")

    model = feedback_model(claim.run.evidence[0].id, [proposal(
        "以后先给结论", content="A different inference after a crash.",
    )])
    settings = Settings(_env_file=None, agent_run_database=runs.path,
                        agent_checkpoint_database=tmp_path / "restart.db")
    resumed_worker = RunWorker(
        settings, repository=runs,
        service_factory=lambda saver, client: DiagnosisService(
            LangChainAgentRunner(model, checkpointer=saver),
            tool_client=FixtureToolClient(
                Telemetry(), Telemetry(), Telemetry()
            ),
            guidance_lookup=forbidden_lookup,
        ),
    )
    assert asyncio.run(resumed_worker.run_once())
    finished = runs.get(state.id)
    assert finished.state.status is RunStatus.SUCCEEDED
    assert finished.state.feedback_updates == original.updates
    assert finished.state.guidance_snapshot == state.guidance_snapshot
    assert library.list_guidance() == []  # Receipt cannot recreate a deletion.
    event = next(e for e in finished.state.events
                 if e.kind == "memory.feedback_saved")
    assert event.data["guidance_ids"] == [str(i) for i in
                                          original.guidance_ids]


def test_inconclusive_source_can_receive_a_human_correction(tmp_path):
    runs = RunRepository(tmp_path / "runs.db")
    source = completed_source(runs, investigation=True)
    # Human confirmation may correct an inconclusive draft with observations.
    with runs.database.transaction() as connection:
        stored = runs.get(source.id)
        stored.state.status = RunStatus.INCONCLUSIVE
        stored.state.diagnosis.status = stored.result.diagnosis.status = (
            DiagnosisStatus.INCONCLUSIVE)
        connection.execute(
            "UPDATE runs SET status=?,state_json=?,result_json=? WHERE id=?",
            ("inconclusive", stored.state.model_dump_json(),
             stored.result.model_dump_json(), str(source.id)),
        )
    message = "已经确认是配置问题，修复后恢复了。"
    learned, _ = execute_feedback(tmp_path, runs, source, message, [proposal(
        message, kind="diagnosis", key=None, content="配置错误，修复后恢复。",
    )])
    assert (
        MemoryRepository(runs.database).list()[0].feedback_run_id
        == learned.state.id
    )


def test_rejected_diagnosis_preview_does_not_log_feedback_quotes():
    message = "私人反馈内容仅保留在受控记忆存储"
    response = ModelResponse(result=[tool_call("DiagnosisProposal", {
        "status": "complete", "root_cause": "Observed failure",
        "confidence": .9, "evidence_ids": [],
        "feedback_updates": [proposal(message).model_dump()],
    })])
    preview, truncated = HarnessMiddleware._candidate_preview(response)
    assert not truncated
    assert "Observed failure" in preview
    assert message not in preview
    assert "feedback_updates" not in preview


def test_pre_feedback_corpus_keeps_its_published_fingerprint():
    snapshot = MemorySnapshot.model_validate_json(
        Path("evals/memory/synthetic_training.json").read_text())
    assert snapshot.fingerprint == (
        "d57757f716b16733ab02ac885904d76d21de471ea488e4cbfbdaec00f81529f8")


def test_guidance_management_and_bounded_context(tmp_path):
    runs = RunRepository(tmp_path / "runs.db")
    source = completed_source(runs, investigation=True)
    learned, _ = execute_feedback(tmp_path, runs, source, "以后先给结论",
                                  [proposal("以后先给结论")])
    library = MemoryRepository(runs.database)
    record = library.list_guidance()[0]

    async def requests():
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as client:
            url = f"/api/v1/guidance/{record.id}"
            assert (await client.get("/api/v1/guidance")).json()[0]["key"] == (
                record.key)
            edited = await client.put(url, json={
                "instruction": "中文" * 900, "expected_revision": 1,
            })
            assert edited.status_code == 200
            assert edited.json()["origin"] == "manual"
            stale = await client.put(url, json={
                "instruction": "stale", "expected_revision": 1,
            })
            assert stale.status_code == 409
            state = learned.state.model_copy(deep=True)
            state.guidance_snapshot = library.guidance_snapshot()
            state.learning_policy.context_bytes = 512
            preview = guidance_context(state)
            assert len(preview.encode()) <= 512
            assert json.loads(preview)["omitted_records"] == 1
            assert (await client.delete(
                f"{url}?expected_revision=2")).status_code == 204
            assert library.guidance_snapshot().records == []

    app.dependency_overrides[get_run_repository] = lambda: runs
    try:
        asyncio.run(requests())
    finally:
        app.dependency_overrides.pop(get_run_repository)


def test_v3_migration_keeps_confirmations_and_old_runs(tmp_path):
    runs = RunRepository(tmp_path / "runs.db")
    source = completed_source(runs)
    original = runs.get(source.id)
    library = MemoryRepository(runs.database)
    memory = library.confirm(confirmation(source))
    with sqlite3.connect(runs.path) as connection:
        connection.execute("DROP TABLE agent_guidance")
        connection.execute("DROP TABLE memory_feedback")
        connection.execute("PRAGMA user_version=3")
    reopened = RunRepository(runs.path)
    assert reopened.get(source.id) == original
    assert MemoryRepository(reopened.database).get(memory.id) == memory
    assert MemoryRepository(reopened.database).list_guidance() == []
