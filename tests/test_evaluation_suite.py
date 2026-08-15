"""Verify replay, outcome scoring, failure retention and batch accounting."""

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

from tracemind.agent import DiagnosisRun, DiagnosisService
from tracemind.agent.errors import ModelConfigurationError
from tracemind.config import Settings
from tracemind.evaluation.evaluator import evaluate_diagnosis
from tracemind.evaluation.replay import (
    ReplayFixture, ReplayTelemetry, run_replay_evaluation,
)
from tracemind.evaluation.runner import (
    run_live_evaluation, run_service_evaluation,
)
from tracemind.evaluation.summary import (
    TokenPricing, build_batch_report, summarize_reports, trace_issues,
)
from tracemind.models import Diagnosis, DiagnosisStatus
from tracemind.models.run import RunStatus


FIXTURES = sorted(Path("evals/replays").glob("*.json"))


def settings(tmp_path: Path) -> Settings:
    """Keep tests independent of local provider credentials and trace files."""
    return Settings(_env_file=None, agent_trace_directory=tmp_path,
                    agent_retry_backoff_seconds=0)


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
        assert report.result.passed
    if path.stem == "transient_recovery":
        assert report.run_state.backend_attempts == 2
        assert report.run_state.tool_calls == {"search_traces": 2}


def test_false_completion_on_insufficient_evidence() -> None:
    fixture = ReplayFixture.load(Path("evals/replays/missing_telemetry.json"))
    run = DiagnosisRun(
        diagnosis=Diagnosis(
            incident_id=fixture.case.incident.id,
            status=DiagnosisStatus.COMPLETE,
            root_cause="Payment Charge failed",
            confidence=.9,
        ), evidence=[], tool_calls={},
    )
    result = evaluate_diagnosis(fixture.case, run, elapsed_ms=1)
    assert result.false_completion
    assert not result.passed


def test_summary_keeps_failures_and_unknown_cost(tmp_path: Path) -> None:
    async def execute():
        return [await run_replay_evaluation(
            ReplayFixture.load(path), settings(tmp_path)
        ) for path in FIXTURES]

    reports = asyncio.run(execute())
    summary = summarize_reports(reports)
    assert summary.runs == summary.regression_passed == 6
    assert summary.passed == 5
    assert summary.failed_runs == 1
    assert summary.trace_valid_runs == 6
    assert summary.retries == summary.recovered_runs == 1
    assert summary.unreported_model_calls == summary.model_calls
    assert summary.estimated_cost_usd is None
    assert summary.priced_runs == 0
    assert summary.false_completions == 0


def test_trace_audit_rejects_isolation_and_missing_terminal(tmp_path: Path):
    fixture = ReplayFixture.load(Path("evals/replays/payment_failure.json"))
    report = asyncio.run(run_replay_evaluation(fixture, settings(tmp_path)))
    state = report.run_state.model_copy(deep=True)
    state.events[0].run_id = uuid4()
    state.events.pop()
    issues = trace_issues(state)
    assert "run_isolation" in issues
    assert "terminal_event" in issues
    state.events[0].sequence = 99
    assert "event_sequence" in trace_issues(state)


def test_estimates_cost_only_for_fully_reported_single_model(tmp_path: Path):
    fixture = ReplayFixture.load(Path("evals/replays/payment_failure.json"))
    report = asyncio.run(run_replay_evaluation(fixture, settings(tmp_path)))
    state = report.run_state
    state.usage.unreported_model_calls = 0
    state.usage.reported_model_calls = state.model_calls
    state.usage.reported_input_tokens = 1000
    state.usage.reported_output_tokens = 500
    rates = {"scripted:scripted-replay": TokenPricing(
        input_per_million=1, output_per_million=2,
    )}
    summary = summarize_reports([report], pricing=rates)
    assert summary.estimated_cost_usd == pytest.approx(.002)
    state.model_switches = 1
    assert (summarize_reports([report], pricing=rates).estimated_cost_usd
            is None)


def test_parallel_replays_keep_separate_runs(tmp_path: Path):
    fixture = ReplayFixture.load(Path("evals/replays/payment_failure.json"))

    async def execute():
        return await asyncio.gather(*[
            run_replay_evaluation(fixture, settings(tmp_path))
            for _ in range(2)
        ])

    reports = asyncio.run(execute())
    assert reports[0].run_state.id != reports[1].run_state.id
    assert all(r.regression_passed for r in reports)
    assert all(not trace_issues(r.run_state) for r in reports)


def test_cancelled_evaluation_retains_terminal_trace(tmp_path: Path):
    fixture = ReplayFixture.load(Path("evals/replays/payment_failure.json"))
    entered = asyncio.Event()

    class WaitingRunner:
        async def run(self, *_args, **_kwargs):
            entered.set()
            await asyncio.Future()

    telemetry = ReplayTelemetry([])
    service = DiagnosisService(telemetry, telemetry, telemetry,
                               WaitingRunner(), event_directory=tmp_path)

    async def execute():
        task = asyncio.create_task(run_service_evaluation(
            fixture.case, service, start=fixture.start, end=fixture.end,
            model_provider="scripted", model_name="waiting",
        ))
        await entered.wait()
        task.cancel()
        return await task

    report = asyncio.run(execute())
    assert report.error_type == "CancelledError"
    assert report.run_state.status is RunStatus.CANCELLED
    assert not trace_issues(report.run_state)
    summary = summarize_reports([report])
    assert summary.cancelled_runs == 1
    assert summary.failed_runs == 0


def test_configuration_failure_is_reported(tmp_path: Path, monkeypatch):
    fixture = ReplayFixture.load(Path("evals/replays/payment_failure.json"))

    def unavailable(_settings):
        raise ModelConfigurationError("synthetic missing key")

    monkeypatch.setattr("tracemind.evaluation.runner.create_chat_model",
                        unavailable)
    report = asyncio.run(run_live_evaluation(
        fixture.case, settings(tmp_path),
        start=fixture.start, end=fixture.end,
    ))
    assert report.error_type == "ModelConfigurationError"
    assert report.run_state is None
    assert summarize_reports([report]).runs_without_state == 1


def test_batch_groups_measurement_modes(tmp_path: Path):
    fixture = ReplayFixture.load(Path("evals/replays/payment_failure.json"))
    scripted = asyncio.run(run_replay_evaluation(fixture, settings(tmp_path)))
    live = scripted.model_copy(deep=True)
    live.execution_mode = "live"
    live.regression_passed = None
    live.model_provider, live.model_name = "test", "real-model"
    live.run_state.id = uuid4()
    batch = build_batch_report([scripted, live])
    assert set(batch.groups) == {
        "scripted_replay/scripted:scripted-replay", "live/test:real-model",
    }
    assert batch.summary.runs == 2
