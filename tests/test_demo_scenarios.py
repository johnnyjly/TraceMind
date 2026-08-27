"""Fault isolation, restoration and CLI behavior for real Demo scenarios."""

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from tracemind.agent import DiagnosisRun
from tracemind.evaluation.demo import (
    SCENARIO_NAMES, activate_flag_config, compose_command,
    prepare_flag_config, restore_flag_config,
)
from tracemind.evaluation.evaluator import evaluate_diagnosis
from tracemind.evaluation.models import EvaluationCase
from tracemind.models import Diagnosis, Evidence


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    """Import a CLI without invoking its entrypoint."""
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def demo(tmp_path):
    """Build a pristine flag file with all three scenario faults off."""
    source = tmp_path / "src/flagd/demo.flagd.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"flags": {
        "paymentFailure": {"defaultVariant": "off",
                           "variants": {"off": 0, "100%": 1}},
        "cartFailure": {"defaultVariant": "off",
                        "variants": {"off": 0, "100%": 1}},
        "intlShippingSlowdown": {"defaultVariant": "off",
                                 "variants": {"off": 0, "5sec": 5}},
    }}))
    return tmp_path


def test_switch_isolates_fault_and_restores_exact_previous_file(demo):
    active, backup = demo / "active.json", demo / "previous.json"
    original = b'{"flags":{"custom":{"defaultVariant":"on"}}}\n'
    active.write_bytes(original)
    inode = active.stat().st_ino
    for scenario in SCENARIO_NAMES:
        case = EvaluationCase.load(ROOT / f"evals/cases/{scenario}.json")
        prepared = prepare_flag_config(
            demo, demo / f"{scenario}.json",
            feature_flag=case.fault.feature_flag, variant=case.fault.variant)
        activate_flag_config(prepared, active, backup)
        flags = json.loads(active.read_text())["flags"]
        assert {name for name, flag in flags.items()
                if flag["defaultVariant"] != "off"} == {
                    case.fault.feature_flag}
        assert active.stat().st_ino == inode
        restore_flag_config(active, backup)
        assert active.read_bytes() == original
        restore_flag_config(active, backup)
        assert active.read_bytes() == original


@pytest.mark.parametrize("scenario", SCENARIO_NAMES[1:])
def test_new_case_requires_evidence_from_actual_cause(scenario):
    case = EvaluationCase.load(ROOT / f"evals/cases/{scenario}.json")
    evidence = Evidence(
        source=case.ground_truth.required_evidence_source,
        service=case.ground_truth.root_cause_service,
        summary="Observed fault", observed_at=datetime.now(UTC))
    diagnosis = Diagnosis(
        incident_id=case.incident.id, status="complete", confidence=.8,
        root_cause=(f"{case.ground_truth.root_cause_service}: "
                    f"{case.ground_truth.root_cause_terms[0]}"),
        evidence_ids=[evidence.id])
    run = DiagnosisRun(diagnosis=diagnosis, evidence=[evidence], tool_calls={})
    assert evaluate_diagnosis(case, run, elapsed_ms=1).passed
    evidence.service = "checkout"
    assert not evaluate_diagnosis(case, run, elapsed_ms=1).passed


@pytest.mark.parametrize("action", ["status", "stop"])
def test_read_and_stop_commands_do_not_change_flags(demo, monkeypatch, action):
    cli = load_script("demo_scenario")
    active = demo / "active.json"
    active.write_text("unchanged")
    monkeypatch.setattr(cli, "GENERATED_FLAG_PATH", active)
    monkeypatch.setattr(cli, "verify_demo_commit", lambda path: None)
    monkeypatch.setattr(cli.shutil, "which", lambda *a, **k: "docker")
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, **kw:
                        calls.append(cmd))
    monkeypatch.setattr("sys.argv", [
        "demo_scenario.py", action, "--demo-path", str(demo)])
    cli.main()
    assert len(calls) == 1
    assert active.read_text() == "unchanged"


def test_prepare_does_not_activate_or_run_docker(demo, monkeypatch):
    cli = load_script("demo_scenario")
    active = demo / "active.json"
    active.write_text("unchanged")
    monkeypatch.setattr(cli, "GENERATED_FLAG_PATH", active)
    monkeypatch.setattr(cli, "CASE_PATH",
                        ROOT / "evals/cases/payment_failure.json")
    monkeypatch.setattr(cli, "verify_demo_commit", lambda path: None)
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw:
                        pytest.fail("prepare must not invoke Docker"))
    monkeypatch.setattr("sys.argv", [
        "demo_scenario.py", "prepare", "--scenario", "shipping_slowdown",
        "--demo-path", str(demo)])
    cli.main()
    assert active.read_text() == "unchanged"
    assert (demo / "shipping_slowdown.flagd.json").exists()


@pytest.mark.parametrize("action", ["switch", "restore"])
def test_switch_recreates_only_flagd(demo, action):
    cmd = compose_command(demo, demo / "override.yaml", action)
    assert cmd[-6:] == [
        "up", "--no-build", "--force-recreate", "--detach", "--no-deps",
        "flagd"]
    assert "--remove-orphans" not in cmd
    assert "--volumes" not in cmd


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_evaluation_default_report_is_specific_to_case(monkeypatch, scenario):
    cli = load_script("run_evaluation")
    outputs = []

    async def fake_run(case, minutes, output, memory):
        """Capture CLI arguments without models or backends."""
        outputs.append(output)
        return SimpleNamespace(result=SimpleNamespace(passed=True))

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr("sys.argv", [
        "run_evaluation.py", "--case", f"evals/cases/{scenario}.json"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert outputs[0].name == scenario.replace("_", "-") + ".json"
