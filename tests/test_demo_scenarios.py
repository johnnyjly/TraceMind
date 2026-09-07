"""Fault isolation, restoration and CLI behavior for real Demo scenarios."""

from datetime import UTC, datetime, timedelta
import importlib.util
import json
from pathlib import Path

import pytest

from tracemind.evaluation.demo import (
    SCENARIO_NAMES,
    activate_flag_config,
    compose_command,
    prepare_flag_config,
    restore_flag_config,
)
from tracemind.evaluation.evaluators import diagnosis_scores
from tracemind.evaluation.models import ExperimentOutput
from tracemind.evaluation.models import EvaluationCase
from tracemind.models import Diagnosis, Evidence
from tracemind.models.result import DiagnosisRun


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
    """Build a pristine flag file with all supported scenario faults off."""
    source = tmp_path / "src/flagd/demo.flagd.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"flags": {
        "paymentFailure": {"defaultVariant": "off",
                           "variants": {"off": 0, "100%": 1}},
        "cartFailure": {"defaultVariant": "off",
                        "variants": {"off": 0, "100%": 1}},
        "intlShippingSlowdown": {"defaultVariant": "off",
                                 "variants": {"off": 0, "5sec": 5}},
        "productCatalogLockContention": {"defaultVariant": "off",
                                         "variants": {"off": False,
                                                      "on": True}},
        "paymentUnreachable": {"defaultVariant": "off",
                               "variants": {"off": False, "on": True}},
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

    def checks():
        output = ExperimentOutput(
            diagnosis=run.diagnosis,
            evidence=run.evidence,
            investigation_start=evidence.observed_at -
            timedelta(
                seconds=1),
            investigation_end=evidence.observed_at +
            timedelta(
                seconds=1))
        return {
            s.name: s.value for s in diagnosis_scores(
                output=output,
                expected_output=case.ground_truth.model_dump(
                    mode="json"))}
    assert checks()["required_source_citation"]
    evidence.service = "frontend"
    assert not checks()["required_source_citation"]


@pytest.mark.parametrize("scenario,misdiagnosis", [
    ("payment_unreachable", "payment rejected the card: Invalid token"),
    ("payment_unreachable", "checkout failed because payment is unavailable"),
    ("product_catalog_lock_contention", "product-catalog is slow"),
])
def test_provenance_scores_do_not_certify_new_fault_causes(
    scenario, misdiagnosis,
):
    case = EvaluationCase.load(ROOT / f"evals/cases/{scenario}.json")
    evidence = Evidence(
        source=case.ground_truth.required_evidence_source,
        service=case.ground_truth.root_cause_service,
        summary="Observed fault", observed_at=datetime.now(UTC))
    diagnosis = Diagnosis(
        incident_id=case.incident.id, status="complete", confidence=.8,
        root_cause=misdiagnosis, evidence_ids=[evidence.id])
    run = DiagnosisRun(diagnosis=diagnosis, evidence=[evidence], tool_calls={})
    output = ExperimentOutput(
        diagnosis=run.diagnosis, evidence=run.evidence,
        investigation_start=evidence.observed_at - timedelta(seconds=1),
        investigation_end=evidence.observed_at + timedelta(seconds=1),
    )
    result = {s.name: s.value for s in diagnosis_scores(
        output=output,
        expected_output=case.ground_truth.model_dump(mode="json"))}
    assert result["citation_integrity"]
    assert "root_cause_correctness" not in result
    assert "false_completion" not in result


@pytest.mark.parametrize("scenario", [
    "product_catalog_lock_contention", "payment_unreachable",
])
def test_switch_new_faults_activates_case_and_preserves_previous(
    demo, monkeypatch, scenario,
):
    cli = load_script("demo_scenario")
    active = demo / "active.flagd.json"
    backup = demo / "previous.flagd.json"
    original = b'{"previous":true}\n'
    active.write_bytes(original)
    monkeypatch.setattr(cli, "GENERATED_FLAG_PATH", active)
    monkeypatch.setattr(cli, "BACKUP_FLAG_PATH", backup)
    monkeypatch.setattr(cli, "verify_demo_commit", lambda path: None)
    monkeypatch.setattr(cli.shutil, "which", lambda *a, **k: "docker")
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, **kw:
                        calls.append(cmd))
    monkeypatch.setattr("sys.argv", [
        "demo_scenario.py", "switch", "--scenario", scenario,
        "--demo-path", str(demo)])
    cli.main()
    case = EvaluationCase.load(ROOT / f"evals/cases/{scenario}.json")
    flags = json.loads(active.read_text())["flags"]
    assert {name for name, flag in flags.items()
            if flag["defaultVariant"] != "off"} == {case.fault.feature_flag}
    assert backup.read_bytes() == original
    assert len(calls) == 1
    assert calls[0][-6:] == [
        "up", "--no-build", "--force-recreate", "--detach", "--no-deps",
        "flagd"]


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
