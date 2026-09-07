"""Harness replay contracts and Langfuse evaluation failure boundaries."""

import asyncio
from pathlib import Path
from uuid import uuid4
import pytest
from replay_support import ReplayFixture, run_replay_evaluation
from tracemind.config import Settings
from tracemind.agent.runtime.audit import trace_issues

FIXTURES = sorted(Path("evals/replays").glob("*.json"))


def settings(tmp_path):
    return Settings(
        _env_file=None,
        agent_trace_directory=tmp_path,
        agent_retry_backoff_seconds=0,
    )


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_replays_follow_expected_outcomes(path: Path, tmp_path: Path) -> None:
    fixture = ReplayFixture.load(path)
    report = asyncio.run(run_replay_evaluation(fixture, settings(tmp_path)))
    assert report.regression_passed
    assert report.execution_mode == "scripted_replay"
    assert report.run_state is not None
    assert trace_issues(report.run_state) == []
    if fixture.expected_stop_reason:
        assert report.error_type == "DiagnosisError"
        assert report.diagnosis is None
        assert report.run_state.stop_reason is fixture.expected_stop_reason
    else:
        assert report.scores["execution_success"]
    if path.stem == "transient_recovery":
        assert report.scores["retries"] == 1
        assert report.scores["recovered"] is True
        assert report.run_state.backend_attempts == 2
        assert report.run_state.tool_calls == {"search_traces": 2}


def test_trace_audit_rejects_isolation_and_missing_terminal(tmp_path):
    report = asyncio.run(
        run_replay_evaluation(
            ReplayFixture.load(FIXTURES[0]), settings(tmp_path)
        )
    )
    state = report.run_state.model_copy(deep=True)
    state.events[0].run_id = uuid4()
    state.events.pop()
    issues = trace_issues(state)
    assert "run_isolation" in issues and "terminal_event" in issues
    state.events[0].sequence = 99
    assert "event_sequence" in trace_issues(state)


def test_parallel_replays_keep_separate_runs(tmp_path):
    fixture = ReplayFixture.load(Path("evals/replays/payment_failure.json"))

    async def execute():
        return await asyncio.gather(
            *[
                run_replay_evaluation(fixture, settings(tmp_path))
                for _ in range(2)
            ]
        )

    reports = asyncio.run(execute())
    assert reports[0].run_state.id != reports[1].run_state.id
    assert all(r.regression_passed for r in reports)


@pytest.mark.parametrize("path", sorted(Path(
    "evals/regressions/live-diagnostic-boundaries").glob("*.json")),
    ids=lambda p: p.stem)
def test_audited_diagnostic_boundaries(path, tmp_path):
    fixture = ReplayFixture.load(path)
    report = asyncio.run(run_replay_evaluation(fixture, settings(tmp_path)))
    assert report.regression_passed
    assert trace_issues(report.run_state) == []
