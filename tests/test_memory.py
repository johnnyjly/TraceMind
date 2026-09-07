"""Check human confirmation, snapshot durability and held-out evaluation."""

import asyncio
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
import pytest
from test_harness import (
    END,
    START,
    ScriptedModel,
    Telemetry,
    diagnosis,
    tool_call,
)
from tool_fixtures import FixtureToolClient

from tracemind.agent.workflow.context import memory_context
from tracemind.agent.memory.lookup import (
    evaluation_memory,
    frozen_memory_lookup,
)
from tracemind.agent.storage.memory_repository import (
    MemoryNotFound,
    MemoryRepository,
)
from tracemind.agent.storage.run_repository import RunRepository
from tracemind.agent.runtime.runner import LangChainAgentRunner
from tracemind.agent.service import DiagnosisService
from tracemind.agent.worker import RunWorker
from tracemind.api.app import app, get_run_repository
from tracemind.api.progress import readable_event
from tracemind.config import Settings
from replay_support import ReplayFixture, run_replay_evaluation
from tracemind.agent.runtime.audit import trace_issues
from tracemind.models import (
    Diagnosis,
    DiagnosisStatus,
    Evidence,
    EvidenceSource,
    Incident,
)
from tracemind.models.memory import (
    MemoryCreate,
    MemoryPolicy,
    MemoryRecord,
    MemoryRevision,
    MemoryScope,
    MemorySnapshot,
)
from tracemind.models.run import (
    RunEvent,
    RunLimits,
    RunPurpose,
    RunState,
    RunStatus,
)
from tracemind.models.storage import RunConflict


SCOPE = MemoryScope(environment="test", deployment_version="v1")
POLICY = MemoryPolicy(scope=SCOPE)
TARGET = Incident(title="checkout errors", description="HTTP 500",
                  affected_services=["checkout"])


def completed_source(runs, *, cited=True, investigation=False):
    """Persist a terminal source with real citation provenance."""
    state = RunState(
        incident=TARGET.model_copy(update={"id": uuid4()}),
        investigation_id=uuid4() if investigation else None,
        investigation_start=START, investigation_end=END,
        limits=RunLimits(), allowed_tools=("query_metrics",),
        status=RunStatus.QUEUED,
    )
    runs.create(state, request_hash=str(state.id),
                new_investigation=investigation)
    claim = runs.claim_next()
    state = claim.run.state
    evidence = Evidence(source=EvidenceSource.LOG, service="payment",
                        observed_at=END, summary="Observed charge failure")
    state.status = RunStatus.SUCCEEDED
    state.diagnosis = Diagnosis(
        incident_id=state.incident.id, status=DiagnosisStatus.COMPLETE,
        root_cause="Charge failed", confidence=.95,
        evidence_ids=[evidence.id] if cited else [],
    )
    runs.finish(state.id, claim.owner, state, [evidence])
    return state


def confirmation(source, **changes):
    """Represent a human's resolution and validation note explicitly."""
    data = {
        **SCOPE.model_dump(), "source_run_id": source.id,
        "services": ["checkout", "payment"], "symptom": "HTTP 500",
        "resolution": "Old network timeout hypothesis",
        "diagnostic_queries": ["Check network errors"],
        "remediation": ["Review network configuration"],
        "confirmed_by": "reviewer", "confirmation_note": "Verified recovery",
        "expires_at": datetime.now(UTC) + timedelta(days=30),
    }
    data.update(changes)
    return MemoryCreate(**data)


def frozen_record(**changes):
    """Create a training record preceding the synthetic held-out window."""
    data = {
        **SCOPE.model_dump(), "source_run_id": uuid4(),
        "services": ["checkout"], "symptom": "Earlier checkout failure",
        "resolution": "Network timeout; do not assume it recurs",
        "confirmed_by": "fixture-reviewer",
        "confirmation_note": "Synthetic training incident only",
        "expires_at": END + timedelta(days=30),
        "source_incident_id": uuid4(), "source_evidence_ids": [uuid4()],
        "confirmed_at": START - timedelta(days=2),
        "updated_at": START - timedelta(days=2),
    }
    data.update(changes)
    return MemoryRecord(**data)


def corpus(records=None):
    """Freeze a corpus before the test incident starts."""
    return MemorySnapshot(scope=SCOPE,
                          selected_at=START - timedelta(days=1),
                          records=records if records is not None
                          else [frozen_record()])


def test_manual_confirmation_scope_lifecycle_and_revision(tmp_path):
    runs = RunRepository(tmp_path / "runs.sqlite3")
    source = completed_source(runs)
    library = MemoryRepository(runs.database)
    assert library.list() == []  # Completion alone never promotes a memory.
    record = library.confirm(confirmation(source))
    assert record.source_incident_id == source.incident.id
    assert record.source_evidence_ids == source.diagnosis.evidence_ids
    now = datetime.now(UTC)
    snapshot = library.snapshot(POLICY, TARGET, now)
    assert snapshot.records == [record]
    assert (
        MemoryRepository((RunRepository(runs.path)).database).get(record.id)
        == record
    )
    for incident in (source.incident,
                     TARGET.model_copy(update={"affected_services": ["cart"]}),
                     TARGET.model_copy(update={"affected_services": []})):
        assert not library.snapshot(POLICY, incident, now).records
    for scope in (MemoryScope(environment="other", deployment_version="v1"),
                  MemoryScope(environment="test", deployment_version="v2")):
        assert not library.snapshot(MemoryPolicy(scope=scope), TARGET,
                                    now).records
    assert not library.snapshot(POLICY, TARGET, record.expires_at).records
    # A source run with saved deployment scope cannot be re-labelled.
    with runs.database.transaction() as connection:
        source.memory_policy = POLICY
        connection.execute("UPDATE runs SET state_json=? WHERE id=?", (
            source.model_dump_json(), str(source.id),
        ))
    with pytest.raises(RunConflict, match="scope"):
        library.confirm(confirmation(source, environment="other"))
    revision = MemoryRevision(**confirmation(
        source, resolution="Corrected payment configuration"
    ).model_dump(), expected_revision=1)
    edited = library.confirm(revision, memory_id=record.id,
                             expected_revision=1)
    assert edited.revision == 2
    assert snapshot.records[0].resolution == record.resolution
    with pytest.raises(RunConflict, match="revision"):
        library.confirm(revision, memory_id=record.id, expected_revision=1)
    revoked = library.revoke(record.id, expected_revision=2)
    assert revoked.revision == 3 and revoked.status == "revoked"
    assert not library.snapshot(POLICY, TARGET, now).records
    with pytest.raises(RunConflict):
        library.delete(record.id, expected_revision=2)
    library.delete(record.id, expected_revision=3)
    with pytest.raises(MemoryNotFound):
        library.get(record.id)
    assert snapshot.records == [record]


def test_confirmation_rejects_unreviewed_or_invalid_provenance(tmp_path):
    runs = RunRepository(tmp_path / "runs.sqlite3")
    source = completed_source(runs, cited=False)
    library = MemoryRepository(runs.database)
    with pytest.raises(RunConflict, match="completed diagnosis"):
        library.confirm(confirmation(source))
    source = completed_source(runs)
    with pytest.raises(RunConflict, match="services"):
        library.confirm(confirmation(source, services=["unknown"]))
    with pytest.raises(RunConflict, match="expiry"):
        library.confirm(confirmation(source, expires_at=START))
    with pytest.raises(ValidationError):
        MemoryCreate(**confirmation(source).model_dump(), confidence=.99)
    with pytest.raises(ValidationError):
        confirmation(source, confirmed_by=" ")
    with pytest.raises(ValidationError):
        confirmation(source, services=["checkout", "checkout"])
    with runs.database.transaction() as connection:
        state = source.model_copy(deep=True)
        state.diagnosis.evidence_ids = [uuid4()]
        connection.execute("UPDATE runs SET state_json=?,result_json=? "
                           "WHERE id=?", (
                               state.model_dump_json(), json.dumps({
                                   "diagnosis": state.diagnosis.model_dump(
                                       mode="json"), "evidence": [],
                                   "tool_calls": {},
                               }), str(source.id),
                           ))
    with pytest.raises(RunConflict, match="citations"):
        library.confirm(confirmation(source))


def test_v2_migration_preserves_existing_run(tmp_path):
    runs = RunRepository(tmp_path / "runs.sqlite3")
    source = completed_source(runs)
    original = runs.get(source.id)
    with sqlite3.connect(runs.path) as connection:
        connection.execute("DROP TABLE incident_memories")
        connection.execute("DROP TABLE agent_guidance")
        connection.execute("DROP TABLE memory_feedback")
        connection.execute("PRAGMA user_version=2")
    reopened = RunRepository(runs.path)
    assert reopened.get(source.id) == original
    assert MemoryRepository(reopened.database).list() == []
    with sqlite3.connect(runs.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
    assert RunRepository(runs.path).get(source.id) == original


def test_frozen_filter_and_prompt_bounds():
    valid = frozen_record()
    invalid = [
        frozen_record(status="revoked"),
        frozen_record(expires_at=START - timedelta(days=1)),
        frozen_record(confirmed_at=START),
        frozen_record(environment="other"),
        frozen_record(deployment_version="v2"),
        frozen_record(services=["cart"]),
        frozen_record(source_incident_id=TARGET.id),
    ]
    snapshot = corpus([valid, *invalid])
    lookup = frozen_memory_lookup(snapshot)
    snapshot.records.clear()  # Caller mutations cannot alter frozen inputs.
    selected = lookup(POLICY, TARGET, datetime.now(UTC))
    assert selected.records == [valid]
    state = RunState(incident=TARGET, investigation_start=START,
                     limits=RunLimits(), allowed_tools=("query_metrics",),
                     investigation_end=END, memory_policy=POLICY,
                     memory_snapshot=selected)
    preview = json.loads(memory_context(state))
    assert preview["records"][0]["bounded_preview"]
    assert str(valid.source_evidence_ids[0]) not in memory_context(state)
    # Unicode scope itself can exceed the smallest context allowance.
    scope = MemoryScope(environment="环" * 120, deployment_version="版" * 120)
    state.memory_policy = MemoryPolicy(scope=scope, context_bytes=512)
    state.memory_snapshot = MemorySnapshot(
        scope=scope, selected_at=START, records=[valid])
    text = memory_context(state)
    assert len(text.encode()) <= 512
    assert json.loads(text)["omitted_records"] == 1


def test_worker_freezes_before_model_and_reuses_snapshot(tmp_path):
    runs = RunRepository(tmp_path / "runs.sqlite3")
    source = completed_source(runs)
    library = MemoryRepository(runs.database)
    record = library.confirm(confirmation(source))
    state = RunState(incident=TARGET, investigation_start=START,
                     limits=RunLimits(), allowed_tools=("query_metrics",),
                     investigation_end=END, memory_policy=POLICY,
                     status=RunStatus.QUEUED)
    runs.create(state, request_hash="with-memory")
    saved = []

    def respond(messages, calls):
        persisted = runs.get(state.id).state
        assert persisted.memory_snapshot.records == [record]
        assert any(e.kind == "memory.selected" for e in persisted.events)
        if calls == 1:
            saved.append(persisted.model_copy(deep=True))
            library.delete(record.id, expected_revision=1)
            return tool_call()
        return diagnosis(messages)

    model = ScriptedModel(respond=respond)
    settings = Settings(_env_file=None, agent_run_database=runs.path,
                        agent_checkpoint_database=tmp_path / "checkpoints.db",
                        agent_trace_directory=tmp_path / "traces")
    # The worker's current settings are off; captured run policy remains on.
    worker = RunWorker(settings, repository=runs,
                       service_factory=lambda saver, client: DiagnosisService(
                           LangChainAgentRunner(model, checkpointer=saver),
                           tool_client=FixtureToolClient(
                               Telemetry(), Telemetry(), Telemetry()
                           ),
                           memory_lookup=library.snapshot,
                       ))
    assert asyncio.run(worker.run_once())
    finished = runs.get(state.id)
    assert finished.result.diagnosis.status is DiagnosisStatus.COMPLETE
    assert finished.state.memory_snapshot.records == [record]
    assert len(finished.evidence) == 1
    assert finished.result.diagnosis.evidence_ids != record.source_evidence_ids
    prompt = model.inputs[0][1].text
    assert "contradictory evidence" in prompt
    assert "Old network timeout hypothesis" in prompt
    assert "memory IDs" in prompt

    def forbidden_lookup(*args):
        raise AssertionError("resume must never requery the library")

    resumed = saved[0]
    assert resumed.memory_snapshot.records == [record]
    service = DiagnosisService(
        LangChainAgentRunner(ScriptedModel(respond=lambda m, c: (
            tool_call() if c == 1 else diagnosis(m)))),
        tool_client=FixtureToolClient(Telemetry(), Telemetry(), Telemetry()),
        memory_lookup=forbidden_lookup,
    )
    run = asyncio.run(service.run_persisted(
        resumed, [], persist=lambda s, e: None,
        is_cancelled=lambda: False, resume=True, graph_resume=False,
    ))
    assert run.state.memory_snapshot == finished.state.memory_snapshot
    assert sum(e.kind == "memory.selected" for e in run.state.events) == 1


def test_disabled_and_explanation_do_not_query_memory(tmp_path):
    def forbidden_lookup(*args):
        raise AssertionError("memory should be disabled")

    service = DiagnosisService(
        LangChainAgentRunner(ScriptedModel(respond=lambda m, c: (
            tool_call() if c == 1 else diagnosis(m)))),
        tool_client=FixtureToolClient(Telemetry(), Telemetry(), Telemetry()),
        memory_lookup=forbidden_lookup,
    )
    run = asyncio.run(service.diagnose_with_evidence(
        TARGET, start=START, end=END))
    assert run.state.memory_snapshot is None
    assert not any(e.kind == "memory.selected" for e in run.state.events)
    runs = RunRepository(tmp_path / "runs.sqlite3")
    source = completed_source(runs, investigation=True)
    follow = runs.follow_up(
        source.investigation_id, message="Explain", purpose=RunPurpose.EXPLAIN,
        limits=RunLimits(), allowed_tools=("read_evidence",),
        request_hash="explain", memory_policy=POLICY,
    )
    assert follow.state.memory_policy is None
    runs.cancel(follow.state.id)
    investigate = runs.follow_up(
        source.investigation_id, message="Check network",
        purpose=RunPurpose.INVESTIGATE, limits=RunLimits(),
        allowed_tools=("query_metrics",), request_hash="investigate",
        memory_policy=POLICY,
    )
    assert investigate.state.memory_policy == POLICY


def test_memory_only_citation_cannot_complete():
    snapshot = corpus()
    record = snapshot.records[0]
    model = ScriptedModel(respond=lambda m, c: tool_call("DiagnosisDraft", {
        "status": "complete", "root_cause": "Network timeout",
        "confidence": .99, "evidence_ids": [str(record.id)],
    }))
    service = DiagnosisService(
        LangChainAgentRunner(model),
        tool_client=FixtureToolClient(Telemetry(), Telemetry(), Telemetry()),
        memory_policy=POLICY,
        memory_lookup=frozen_memory_lookup(snapshot),
    )
    run = asyncio.run(service.diagnose_with_evidence(
        TARGET, start=START, end=END))
    assert run.diagnosis.status is DiagnosisStatus.INCONCLUSIVE
    assert run.evidence == []
    assert not run.diagnosis.evidence_ids


def test_frozen_memory_evaluation_is_paired_and_never_uses_live_db(tmp_path):
    fixture = ReplayFixture.load(Path("evals/replays/payment_failure.json"))
    snapshot = corpus()
    settings = Settings(_env_file=None, agent_trace_directory=tmp_path,
                        agent_memory_enabled=True,
                        agent_memory_environment=SCOPE.environment,
                        agent_memory_deployment_version="v1",
                        agent_run_database=tmp_path / "must-not-exist.db")

    async def compare():
        off = await run_replay_evaluation(fixture, settings)
        on = await run_replay_evaluation(fixture, settings,
                                         memory_snapshot=snapshot)
        return off, on

    off, on = asyncio.run(compare())
    assert off.regression_passed and on.regression_passed
    assert off.memory_mode == "off" and off.run_state.memory_snapshot is None
    assert on.memory_mode == "frozen"
    assert on.memory_corpus_id == snapshot.fingerprint
    assert on.run_state.memory_snapshot.records == snapshot.records
    assert on.run_state.limits == off.run_state.limits
    assert on.run_state.tool_calls == off.run_state.tool_calls
    assert not settings.agent_run_database.exists()
    assert off.run_state.id != on.run_state.id
    corrupted = on.run_state.model_copy(deep=True)
    corrupted.memory_snapshot.records[0].resolution = "Edited after execution"
    assert "memory_snapshot" in trace_issues(corrupted)
    for path in ("missing_telemetry", "conflicting_evidence"):
        negative = ReplayFixture.load(Path(f"evals/replays/{path}.json"))
        report = asyncio.run(run_replay_evaluation(
            negative, settings,
            memory_snapshot=snapshot))
        assert report.regression_passed
        assert (report.diagnosis.status
                == negative.case.ground_truth.expected_status)
    with pytest.raises(ValueError, match="before evaluation"):
        evaluation_memory(snapshot.model_copy(update={"selected_at": END}),
                          POLICY, start=START, max_records=3,
                          context_bytes=4096)
    wrong = MemoryPolicy(scope=MemoryScope(environment="wrong",
                                           deployment_version="v1"))
    with pytest.raises(ValueError, match="scope"):
        evaluation_memory(snapshot, wrong, start=START, max_records=3,
                          context_bytes=4096)
    stale = corpus([frozen_record(expires_at=START)])
    policy, lookup = evaluation_memory(stale, POLICY, start=START,
                                       max_records=3, context_bytes=4096)
    assert not lookup(policy, TARGET, datetime.now(UTC)).records
    with pytest.raises(ValidationError):
        MemorySnapshot(**snapshot.model_dump(), ground_truth={})


def test_memory_configuration_requires_explicit_deployment():
    assert Settings(_env_file=None).memory_policy is None
    with pytest.raises(ValidationError, match="environment"):
        Settings(_env_file=None, agent_memory_enabled=True)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, agent_memory_enabled=True,
                 agent_memory_environment=" ",
                 agent_memory_deployment_version="v1")


def test_memory_management_api_and_safe_progress(tmp_path):
    runs = RunRepository(tmp_path / "runs.sqlite3")
    source = completed_source(runs)

    async def requests():
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as client:
            url = "/api/v1/memories"
            body = confirmation(source).model_dump(mode="json")
            response = await client.post(url, json=body)
            assert response.status_code == 201, response.text
            record = response.json()
            target = f"{url}/{record['id']}"
            assert (await client.get(target)).json() == record
            assert len((await client.get(url)).json()) == 1
            exported = await client.get(f"{url}/snapshot", params={
                "environment": "test", "deployment_version": "v1",
                "service": "checkout",
            })
            assert exported.status_code == 200
            assert exported.json()["records"] == [record]
            blank = await client.get(f"{url}/snapshot", params={
                "environment": " ", "deployment_version": "v1",
                "service": "checkout",
            })
            assert blank.status_code == 422
            updated = await client.put(target, json={
                **body, "expected_revision": 1, "resolution": "Correction",
            })
            assert updated.status_code == 200
            stale = await client.put(target, json={
                **body, "expected_revision": 1,
            })
            assert stale.status_code == 409
            revoked = await client.post(f"{target}/revoke?expected_revision=2")
            assert revoked.json()["status"] == "revoked"
            assert (await client.delete(
                f"{target}?expected_revision=3")).status_code == 204
            assert (await client.get(target)).status_code == 404
            assert (await client.post(url, json={
                **body, "source_run_id": str(uuid4()),
            })).status_code == 404

    app.dependency_overrides[get_run_repository] = lambda: runs
    try:
        asyncio.run(requests())
    finally:
        app.dependency_overrides.pop(get_run_repository)
    event = RunEvent(run_id=uuid4(), span_id=uuid4(), sequence=1,
                     kind="memory.selected",
                     data={"record_count": 2, "resolution": "private"})
    progress = readable_event(event)
    assert progress.phase == "historical_context"
    assert progress.data == {"record_count": 2}
