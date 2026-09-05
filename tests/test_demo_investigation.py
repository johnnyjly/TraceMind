"""Check that the interactive CLI distinguishes completion from acceptance."""

import asyncio
import importlib.util
import logging
from pathlib import Path

from tracemind.config import Settings
from tracemind.evaluation.replay import ReplayFixture, run_replay_evaluation
from tracemind.models.result import DiagnosisRun


def test_result_reports_actual_budget_and_failed_case_score(tmp_path, caplog):
    spec = importlib.util.spec_from_file_location(
        "demo_investigation", Path("scripts/demo_investigation.py"))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    fixture = ReplayFixture.load(Path("evals/replays/payment_failure.json"))
    report = asyncio.run(run_replay_evaluation(
        fixture, Settings(_env_file=None, agent_trace_directory=tmp_path)))
    run = DiagnosisRun(
        diagnosis=report.diagnosis, evidence=report.evidence,
        tool_calls={"search_traces": 19}, state=report.run_state,
    )
    run.state.limits.tool_calls = 25
    run.state.tool_calls = run.tool_calls
    with caplog.at_level(logging.INFO):
        cli.log_result(run.model_dump(mode="json"), fixture.case)
    assert "Execution: succeeded (completed)" in caplog.text
    assert "tool calls 19/25" in caplog.text
    assert "failed scoring (case tool budget: 12)" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO):
        cli.log_result(run.model_dump(mode="json"))
    assert "Case " not in caplog.text
