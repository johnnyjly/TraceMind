"""Pinned source access, provenance boundaries and retained R citations."""

import asyncio
from datetime import UTC, datetime, timedelta
import json
import subprocess

import pytest
from langchain_core.messages import AIMessage

from tracemind.agent.runtime.harness import RunHarness
from tracemind.agent.runtime.sources import create_source_tools
from tracemind.agent.runtime.tools import EvidenceCollector
from tracemind.agent.runtime.recovery import ToolUnavailable
from tracemind.agent.storage.run_repository import RunRepository
from tracemind.agent.verification import (
    parse_review,
    review_packet,
    subject_for,
    subject_hash,
    validate_review,
)
from tracemind.agent.workflow.references import register_references
from tracemind.agent.workflow.validation import resolve_proposal
from tracemind.models import Evidence, EvidenceSource, Incident, Diagnosis
from tracemind.models.result import DiagnosisProposal, ProposalStatement
from tracemind.models.run import RunLimits, RunState, RunPurpose
from tracemind.models.sources import SOURCE_TOOLS, SourcePolicy
from tracemind.sources.provider import SourceProvider

AT = datetime(2026, 1, 1, 12, tzinfo=UTC)


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


@pytest.fixture
def source_case(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "app.js").write_text(
        'const currency = input.currency || "USD";\nreturn'
        " currency.toUpperCase();\n"
    )
    git(repo, "add", "app.js")
    git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "old",
    )
    old = git(repo, "rev-parse", "HEAD")
    (repo / "app.js").write_text(
        "const currency = input.currency;\nreturn currency.toUpperCase();\n"
    )
    (repo / "link").symlink_to("/etc/passwd")
    (repo / "binary").write_bytes(b"\x00\x01")
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "new",
    )
    new = git(repo, "rev-parse", "HEAD")
    releases = []
    for index, commit in enumerate((old, new), 1):
        image = "sha256:" + str(index) * 64
        releases.append(
            dict(
                id=f"v{index}",
                service="quote",
                repository_id="demo",
                commit=commit,
                version=f"v{index}",
                scope={"environment": "lab"},
                deployed_at=(AT - timedelta(hours=3 - index)).isoformat(),
                instances=[f"instance-{index}"],
                image_id=image,
                previous_release_id="v1" if index == 2 else None,
                provenance="test build capture",
                captured_at=AT.isoformat(),
                build=dict(
                    record_id=f"build-{index}",
                    captured_at=AT.isoformat(),
                    producer="unit-test",
                    source_tree=git(repo, "rev-parse", commit + "^{tree}"),
                    image_id=image,
                ),
            )
        )
    policy = SourcePolicy.model_validate(
        dict(
            repositories=[
                dict(
                    id="demo",
                    service="quote",
                    path=str(repo),
                    runtime_paths={"/app": "."},
                )
            ],
            releases=releases,
        )
    )
    observation = Evidence(
        source=EvidenceSource.TRACE,
        summary="quote exception",
        observed_at=AT,
        service="quote",
        attributes={
            "resource": {
                "service.version": "v2",
                "service.instance.id": "instance-2",
                "deployment.environment.name": "lab",
            },
            "is_error": True,
        },
    )
    return repo, policy, observation


def state_for(policy, evidence, **changes):
    return RunState(
        incident=Incident(
            title="quote fails", description="Inspect quote failure"
        ),
        investigation_start=AT - timedelta(minutes=1),
        investigation_end=AT + timedelta(minutes=1),
        limits=RunLimits(tool_calls=12),
        allowed_tools=SOURCE_TOOLS,
        source_policy=policy,
        evidence_ids=[evidence.id],
        **changes,
    )


def test_resolve_read_and_diff_are_pinned_and_roundtrip(source_case, tmp_path):
    repo, policy, evidence = source_case
    restored = SourcePolicy.model_validate_json(policy.model_dump_json())
    assert restored.snapshot_id == policy.snapshot_id

    async def exercise():
        provider = SourceProvider(restored)
        release = await provider.resolve(evidence)
        assert release.mapping_status == "matched"
        (repo / "app.js").write_text("uncommitted content must not be read")
        source = await provider.read(release, "/app/app.js", 1, 200)
        assert (
            "input.currency;" in source.content
            and "uncommitted" not in source.content
        )
        diff = await provider.diff(release, "app.js")
        assert '-const currency = input.currency || "USD"' in diff.content
        assert diff.metadata["base_commit"] == policy.releases[0].commit
        state = state_for(policy, evidence)
        state.sources = [release, source, diff]
        saved = RunRepository(tmp_path / "runs.db")
        from tracemind.models.run import RunStatus

        state.status = RunStatus.QUEUED
        saved.create(state, request_hash="source")
        reread = saved.get(state.id).state
        assert [r.id for r in reread.sources] == [r.id for r in state.sources]
        assert reread.source_policy.snapshot_id == policy.snapshot_id

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "variation,expected",
    [
        ("missing", "missing"),
        ("declared", "declared"),
        ("ambiguous", "ambiguous"),
        ("wrong_instance", "mismatch"),
        ("wrong_tree", "mismatch"),
        ("before_release", "missing"),
        ("missing_commit", "unavailable"),
    ],
)
def test_mapping_never_guesses_latest(source_case, variation, expected):
    _, policy, evidence = source_case
    if variation == "missing":
        policy.releases = []
    elif variation == "declared":
        policy.releases[1].build = None
    elif variation == "ambiguous":
        policy.releases.append(
            policy.releases[1].model_copy(update={"id": "duplicate"})
        )
    elif variation == "wrong_instance":
        evidence.attributes["resource"]["service.instance.id"] = "other"
    elif variation == "wrong_tree":
        policy.releases[1].build.source_tree = "f" * 40
    elif variation == "before_release":
        evidence.observed_at = AT - timedelta(days=1)
    elif variation == "missing_commit":
        policy.releases[1].commit = "f" * 40
    result = asyncio.run(SourceProvider(policy).resolve(evidence))
    assert result.mapping_status == expected


@pytest.mark.parametrize(
    "path", ["../outside", "/etc/passwd", "link", "binary", "missing"]
)
def test_source_path_failures_are_explicit(source_case, path):
    _, policy, evidence = source_case

    async def exercise():
        provider = SourceProvider(policy)
        resolved = await provider.resolve(evidence)
        with pytest.raises(ValueError):
            await provider.read(resolved, path, 1, 100)

    asyncio.run(exercise())


def test_tools_share_budget_and_explain_reads_without_git(
    source_case, monkeypatch
):
    _, policy, evidence = source_case
    state = state_for(policy, evidence)
    collector = EvidenceCollector(initial=[evidence])
    harness = RunHarness(state)
    register_references(state)

    async def exercise():
        tools = {t.name: t for t in create_source_tools(harness, collector)}
        first = await tools["resolve_deployment"].ainvoke(
            {"evidence_ref": "E1"}
        )
        assert (
            first[0]["id"] == "R1" and first[0]["mapping_status"] == "matched"
        )
        result = await tools["read_source"].ainvoke(
            {"deployment_ref": "R1", "path": "app.js"}
        )
        assert result[0]["id"] == "R2"
        assert sum(state.tool_calls.values()) == 2
        assert harness.state.backend_attempts == 0
        state.purpose = RunPurpose.EXPLAIN

        def fail(*args, **kwargs):
            raise AssertionError("explain opened source")

        monkeypatch.setattr(SourceProvider, "read", fail)
        tools = {t.name: t for t in create_source_tools(harness, collector)}
        assert set(tools) == {"read_source_reference"}
        page = await tools["read_source_reference"].ainvoke(
            {"source_ref": "R2"}
        )
        assert "toUpperCase" in page[0]["content"]

    asyncio.run(exercise())


def test_source_budget_and_new_window_do_not_reuse_mapping(source_case):
    _, policy, evidence = source_case
    policy.max_calls = 2
    state = state_for(policy, evidence)
    harness = RunHarness(state)
    collector = EvidenceCollector(initial=[evidence])
    register_references(state)

    async def exercise():
        tools = {t.name: t for t in create_source_tools(harness, collector)}
        await tools["resolve_deployment"].ainvoke({"evidence_ref": "E1"})
        state.investigation_end += timedelta(minutes=1)
        with pytest.raises(ToolUnavailable):
            await tools["read_source"].ainvoke(
                {"deployment_ref": "R1", "path": "app.js"}
            )
        with pytest.raises(ToolUnavailable) as error:
            await tools["read_source_reference"].ainvoke({"source_ref": "R1"})
        assert error.value.failure.category == "source_budget_exhausted"
        event = next(
            e
            for e in reversed(state.events)
            if e.kind == "sources.unavailable"
        )
        assert event.data["category"] == "source_budget_exhausted"
        candidate = Diagnosis(
            incident_id=state.incident.id,
            status="inconclusive",
            root_cause="Source unavailable.",
            confidence=0.1,
            evidence_ids=[evidence.id],
        )
        packet = review_packet(subject_for(state, candidate, [evidence]))
        assert any(
            e["kind"] == "sources.unavailable"
            and e["data"]["category"] == "source_budget_exhausted"
            for e in packet["query_events"]
        )

    asyncio.run(exercise())


def test_source_fact_citations_review_and_hash(source_case):
    _, policy, evidence = source_case
    state = state_for(policy, evidence)
    state.sources = [asyncio.run(SourceProvider(policy).resolve(evidence))]
    register_references(state)
    proposal = DiagnosisProposal(
        status="inconclusive",
        confidence=0.2,
        findings=[
            ProposalStatement(
                text="Release record identifies v2.", source_refs=["R1"]
            )
        ],
        hypotheses=[],
        limitations=[],
        proposed_checks=[],
    )
    draft = resolve_proposal(proposal, state, [evidence])
    candidate = Diagnosis(
        incident_id=state.incident.id,
        **draft.model_dump(exclude={"feedback_updates"}),
    )
    subject = subject_for(state, candidate, [evidence]).model_copy(
        update={"purpose": None}
    )
    packet = review_packet(subject)
    assert packet["sources"][0]["ref"] == "R1"
    assert packet["items"][0]["source_refs"] == ["R1"]
    response = AIMessage(
        content=json.dumps(
            dict(
                assessments=[
                    dict(
                        item_id="F1",
                        kind="source_fact",
                        verdict="supported",
                        supporting_source_refs=["R1"],
                        rationale="Recorded release identity.",
                    )
                ],
                root_cause_correctness=None,
                reference_assessment=None,
                rationale="source fact",
                completion=None,
            )
        )
    )
    review = parse_review(response, subject)
    assert review.claims[0].supporting_source_refs == [state.sources[0].id]
    review.claims[0].kind = "diagnostic"
    with pytest.raises(ValueError, match="diagnostic_support"):
        validate_review(review, subject)
    review.claims[0].supporting_ids = [evidence.id]
    validate_review(review, subject)
    before = subject_hash(subject)
    changed = subject.model_copy(deep=True)
    changed.sources[0].content += " changed"
    # Any mutation invalidates the cited identity, never reuses the old review.
    with pytest.raises(KeyError):
        subject_hash(changed)
    assert before


def test_api_admission_freezes_catalog(source_case, tmp_path):
    import httpx
    from tracemind.api.app import (
        create_app,
        get_request_settings,
        get_run_repository,
    )
    from tracemind.config import Settings

    _, policy, evidence = source_case
    manifest = tmp_path / "releases.json"
    manifest.write_text(policy.model_dump_json())
    repository = RunRepository(tmp_path / "runs.sqlite3")
    app = create_app()
    app.dependency_overrides[get_request_settings] = lambda: Settings(
        _env_file=None,
        agent_source_catalog=manifest,
        agent_allowed_tools=SOURCE_TOOLS,
    )
    app.dependency_overrides[get_run_repository] = lambda: repository

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            state = state_for(policy, evidence)
            response = await client.post(
                "/api/v1/runs",
                json={
                    "incident": state.incident.model_dump(mode="json"),
                    "investigation_start": (
                        state.investigation_start.isoformat()
                    ),
                    "investigation_end": state.investigation_end.isoformat(),
                },
            )
            assert response.status_code == 202, response.text
            manifest.unlink()
            from uuid import UUID

            frozen = repository.get(UUID(response.json()["id"])).state
            assert frozen.source_policy.snapshot_id == policy.snapshot_id

    asyncio.run(exercise())


def test_scope_unknown_and_conflicting_runtime_identity_are_not_matched(
    source_case,
):
    from tracemind.models.intake import ContextScope

    _, policy, evidence = source_case
    evidence.attributes["resource"].pop("deployment.environment.name")
    result = asyncio.run(
        SourceProvider(policy).resolve(
            evidence, ContextScope(environment="lab")
        )
    )
    assert result.mapping_status == "missing"
    assert result.metadata["reason"] == "scope_unknown"
    evidence.attributes["resource_metadata"] = {
        "conflicts": {"service.instance.id": []}
    }
    result = asyncio.run(SourceProvider(policy).resolve(evidence))
    assert result.mapping_status == "declared"


def test_reference_cap_counts_escaped_content_and_metadata(source_case):
    from tracemind.models.sources import bound_reference

    _, policy, evidence = source_case
    policy.response_bytes = 1024
    result = SourceProvider(policy).reference(
        "deployment", policy.releases[1], '"\\中文' * 3000
    )
    assert len(result.model_dump_json().encode()) <= 1024
    assert result.truncated
    result.metadata["too_large"] = "x" * 2000
    with pytest.raises(ValueError, match="metadata"):
        bound_reference(result, 1024)


def test_review_links_source_to_deployment_to_runtime(source_case):
    from tracemind.agent.verification import review_schema

    _, policy, evidence = source_case

    async def references():
        provider = SourceProvider(policy)
        deployment = await provider.resolve(evidence)
        source = await provider.read(deployment, "app.js", 1, 2)
        return [deployment, source]

    state = state_for(policy, evidence)
    state.sources = asyncio.run(references())
    diagnosis = Diagnosis(
        incident_id=state.incident.id,
        status="inconclusive",
        summary="Investigation incomplete.",
        root_cause="Unresolved",
        confidence=0.1,
        evidence_ids=[evidence.id],
    )
    packet = review_packet(subject_for(state, diagnosis, [evidence]))
    assert packet["sources"][0]["evidence_ids"] == ["E1"]
    assert packet["sources"][1]["metadata"]["deployment_ref"] == "R1"
    fields = review_schema(packet)["$defs"]["ItemAssessment"]["properties"]
    assert fields["supporting_source_refs"]["items"]["enum"] == ["R1", "R2"]


def test_new_run_cannot_reuse_old_deployment_even_in_same_window(source_case):
    from uuid import uuid4

    _, policy, evidence = source_case
    state = state_for(policy, evidence)
    harness = RunHarness(state)
    register_references(state)

    async def exercise():
        tools = {
            t.name: t
            for t in create_source_tools(
                harness, EvidenceCollector(initial=[evidence])
            )
        }
        await tools["resolve_deployment"].ainvoke({"evidence_ref": "E1"})
        state.id = uuid4()
        with pytest.raises(ToolUnavailable):
            await tools["read_source"].ainvoke(
                {"deployment_ref": "R1", "path": "app.js"}
            )

    asyncio.run(exercise())


def test_followup_rejects_budget_smaller_than_retained_sources(source_case):
    from uuid import uuid4
    from tracemind.agent.workflow.followup import plan_follow_up
    from tracemind.models.storage import StoredRun, RunConflict

    _, policy, evidence = source_case
    state = state_for(policy, evidence)
    state.sources = [asyncio.run(SourceProvider(policy).resolve(evidence))]
    parent = StoredRun(
        state=state,
        evidence=[evidence],
        result=None,
        cancel_requested=False,
        attempt=0,
        lease_until=None,
    )
    with pytest.raises(RunConflict, match="observation budget"):
        plan_follow_up(
            parent,
            [],
            investigation_id=uuid4(),
            message="Explain",
            purpose=RunPurpose.EXPLAIN,
            limits=RunLimits(observation_bytes=100),
            allowed_tools=SOURCE_TOOLS,
            start=None,
            end=None,
            memory_policy=None,
        )


def test_saved_verification_preserves_empty_source_snapshot(source_case):
    from tracemind.evaluation.models import ExperimentOutput

    _, policy, evidence = source_case
    state = state_for(policy, evidence)
    diagnosis = Diagnosis(
        incident_id=state.incident.id,
        status="inconclusive",
        summary="Investigation incomplete.",
        root_cause="Unresolved",
        confidence=0.1,
        evidence_ids=[evidence.id],
    )
    subject = subject_for(state, diagnosis, [evidence])
    output = ExperimentOutput(
        diagnosis=diagnosis, verification_subject=subject
    )
    saved = ExperimentOutput.from_saved(output.model_dump(mode="json"))
    assert subject_hash(saved.verification_subject) == subject_hash(subject)
    assert review_packet(saved.verification_subject)["version"].endswith(
        "sources-v1"
    )


def test_private_key_middle_lines_are_redacted_before_source_window(
    source_case, monkeypatch
):
    import tracemind.sources.provider as module

    _, policy, evidence = source_case
    provider = SourceProvider(policy)
    deployment = asyncio.run(provider.resolve(evidence))
    body = (
        "first\n-----BEGIN PRIVATE KEY-----\n"
        "synthetic-not-a-real-secret\n-----END PRIVATE KEY-----\nlast\n"
    ).encode()

    async def fake_git(repo, *args, **kwargs):
        if args[0] == "ls-tree":
            return b"100644 blob " + b"a" * 40 + b"\tapp.js\x00"
        if args[1] == "-s":
            return str(len(body)).encode()
        return body

    monkeypatch.setattr(module, "git", fake_git)
    result = asyncio.run(provider.read(deployment, "app.js", 3, 5))
    assert "synthetic-not-a-real-secret" not in result.content
    assert "3: [REDACTED]" in result.content
    assert "5: last" in result.content
    assert result.redacted and result.metadata["end_line"] == 5


def test_source_line_that_cannot_fit_is_unavailable(source_case, monkeypatch):
    import tracemind.sources.provider as module

    _, policy, evidence = source_case
    provider = SourceProvider(policy)
    deployment = asyncio.run(provider.resolve(evidence))
    body = b"x" * 40000

    async def fake_git(repo, *args, **kwargs):
        if args[0] == "ls-tree":
            return b"100644 blob " + b"a" * 40 + b"\tapp.js\x00"
        if args[1] == "-s":
            return str(len(body)).encode()
        return body

    monkeypatch.setattr(module, "git", fake_git)
    with pytest.raises(ValueError, match="line exceeds"):
        asyncio.run(provider.read(deployment, "app.js", 1, 200))


def test_source_page_never_returns_an_unchanged_cursor(source_case):
    from tracemind.models.sources import SourceReference

    _, policy, evidence = source_case
    policy.response_bytes = 1024
    state = state_for(policy, evidence, purpose=RunPurpose.EXPLAIN)
    state.sources = [
        SourceReference(
            kind="deployment",
            snapshot_id=policy.snapshot_id,
            service="quote",
            mapping_status="missing",
            content="x" * 100,
            metadata={"large": "y" * 400},
        )
    ]
    register_references(state)
    tools = {
        t.name: t
        for t in create_source_tools(
            RunHarness(state), EvidenceCollector(initial=[evidence])
        )
    }

    async def read():
        try:
            page = await tools["read_source_reference"].ainvoke(
                {"source_ref": "R1"}
            )
        except ToolUnavailable:
            return  # Explicit refusal is preferable to a non-advancing page.
        assert page[0]["content"] and page[0]["next_offset"] != 0

    asyncio.run(read())


def test_diff_checks_credentials_outside_returned_hunk(
    source_case, monkeypatch
):
    import tracemind.sources.provider as module

    _, policy, evidence = source_case
    provider = SourceProvider(policy)
    deployment = asyncio.run(provider.resolve(evidence))
    body = (
        "-----BEGIN PRIVATE KEY-----\n"
        + "synthetic-line\n" * 20
        + "-----END PRIVATE KEY-----\n"
    ).encode()

    async def fake_git(repo, *args, **kwargs):
        if args[0] == "ls-tree":
            return b"100644 blob " + b"a" * 40 + b"\tapp.js\x00"
        if args[0] == "diff":
            return (
                b":100644 100644 "
                + b"a" * 40
                + b" "
                + b"b" * 40
                + b" M\x00app.js\x00"
            )
        if args[1] == "-s":
            return str(len(body)).encode()
        if args[1] == "blob":
            return body
        raise AssertionError("unsafe diff should not be emitted")

    monkeypatch.setattr(module, "git", fake_git)
    with pytest.raises(ValueError, match="credential-bearing"):
        asyncio.run(provider.diff(deployment, "app.js"))


def test_diff_missing_path_is_not_evidence_of_no_change(source_case):
    from tracemind.sources.provider import SourcePathNotFound

    _, policy, evidence = source_case

    async def exercise():
        provider = SourceProvider(policy)
        deployment = await provider.resolve(evidence)
        with pytest.raises(SourcePathNotFound):
            await provider.diff(deployment, "release-fixture/app.js")
        actual = await provider.diff(deployment, "/app/app.js")
        assert '-const currency = input.currency || "USD";' in actual.content
        assert actual.metadata["path"] == "app.js"

    asyncio.run(exercise())
