from datetime import UTC, datetime
import json
import os
from pathlib import Path

import pytest

from tracemind.evaluation.demo import (
    compose_command,
    docker_environment,
    prepare_flag_config,
)
from tracemind.evaluation.evaluators import diagnosis_scores
from tracemind.evaluation.models import ExperimentOutput
from tracemind.evaluation.models import EvaluationCase
from tracemind.models import (
    Diagnosis,
    DiagnosisStatus,
    Evidence,
    EvidenceSource,
)
from tracemind.models.result import DiagnosisRun


CASE_PATH = Path("evals/cases/payment_failure.json")


@pytest.mark.parametrize('explicit_status', [False, True])
def test_case_cli_does_not_add_an_implicit_completion_requirement(
        monkeypatch, tmp_path, explicit_status):
    import runpy
    import sys
    from types import SimpleNamespace
    from tracemind.config import Settings

    payload = json.loads(CASE_PATH.read_text())
    if explicit_status:
        payload['ground_truth']['expected_status'] = 'complete'
    case_path = tmp_path / 'case.json'
    case_path.write_text(json.dumps(payload))
    main = runpy.run_path('scripts/run_experiment.py')['main']
    captured = []
    client = SimpleNamespace(auth_check=lambda: True, flush=lambda: None)

    def experiment(*args, **kwargs):
        captured.append(args[2])
        return SimpleNamespace(format=lambda: 'offline', experiment_id='local')

    monkeypatch.setattr(sys, 'argv', [
        'run_experiment.py', '--task', 'agent', '--name', 'offline',
        '--case', str(case_path), '--max-model-calls', '1'])
    monkeypatch.setitem(main.__globals__, 'Settings',
                        lambda **kwargs: Settings(_env_file=None))
    monkeypatch.setitem(main.__globals__, 'client_for',
                        lambda *args, **kwargs: client)
    monkeypatch.setitem(main.__globals__, 'run_experiment', experiment)
    monkeypatch.setitem(main.__globals__, 'accepted', lambda result: True)
    monkeypatch.setitem(main.__globals__, 'verify_publication',
                        lambda *args, **kwargs: True)
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 0
    expected = captured[0][0]['expected_output']
    assert ('expected_status' in expected) is explicit_status
    assert expected['root_cause_service'] == 'payment'


def test_payment_failure_case_accepts_grounded_payment_diagnosis() -> None:
    case = EvaluationCase.load(CASE_PATH)
    evidence = Evidence(
        source=EvidenceSource.TRACE,
        summary="Error span in payment: Charge",
        observed_at=datetime(2024, 9, 1, tzinfo=UTC),
        service="payment",
    )
    run = DiagnosisRun(
        diagnosis=Diagnosis(
            incident_id=case.incident.id,
            status=DiagnosisStatus.COMPLETE,
            root_cause="The payment Charge operation failed.",
            confidence=0.9,
            evidence_ids=[evidence.id],
        ),
        evidence=[evidence],
        tool_calls={"search_traces": 1, "get_trace_detail": 1},
    )

    result = {score.name: score.value for score in diagnosis_scores(
        output=ExperimentOutput(diagnosis=run.diagnosis, evidence=run.evidence,
                                investigation_start=datetime(
                                    2024, 9, 1, tzinfo=UTC),
                                investigation_end=datetime(
                                    2024, 9, 2, tzinfo=UTC)),
        expected_output=case.ground_truth.model_dump(mode="json"))}

    assert result["execution_success"] is True
    assert result["required_source_citation"] is True
    assert result["citation_integrity"] is True


def test_payment_failure_case_rejects_checkout_only_evidence() -> None:
    case = EvaluationCase.load(CASE_PATH)
    evidence = Evidence(
        source=EvidenceSource.LOG,
        summary="Checkout returned HTTP 500",
        observed_at=datetime(2024, 9, 1, tzinfo=UTC),
        service="checkout",
    )
    run = DiagnosisRun(
        diagnosis=Diagnosis(
            incident_id=case.incident.id,
            status=DiagnosisStatus.COMPLETE,
            root_cause="Checkout failed with an error.",
            confidence=0.8,
            evidence_ids=[evidence.id],
        ),
        evidence=[evidence],
        tool_calls={"search_logs": 1},
    )

    result = {score.name: score.value for score in diagnosis_scores(
        output=ExperimentOutput(diagnosis=run.diagnosis, evidence=run.evidence,
                                investigation_start=datetime(
                                    2024, 9, 1, tzinfo=UTC),
                                investigation_end=datetime(
                                    2024, 9, 2, tzinfo=UTC)),
        expected_output=case.ground_truth.model_dump(mode="json"))}

    assert result["required_source_citation"] is False
    assert "root_cause_correctness" not in result


def test_prepare_flag_config_changes_only_selected_default(
    tmp_path: Path,
) -> None:
    demo_path = tmp_path / "demo"
    source_path = demo_path / "src" / "flagd" / "demo.flagd.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        json.dumps(
            {
                "flags": {
                    "paymentFailure": {
                        "defaultVariant": "off",
                        "variants": {"off": 0, "100%": 1},
                    },
                    "otherFlag": {
                        "defaultVariant": "off",
                        "variants": {"off": False, "on": True},
                    },
                }
            }
        )
    )

    output = prepare_flag_config(
        demo_path,
        tmp_path / "generated.json",
        feature_flag="paymentFailure",
        variant="100%",
    )
    generated = json.loads(output.read_text())

    assert generated["flags"]["paymentFailure"]["defaultVariant"] == "100%"
    assert generated["flags"]["otherFlag"]["defaultVariant"] == "off"
    assert json.loads(source_path.read_text())["flags"]["paymentFailure"][
        "defaultVariant"
    ] == "off"


def test_compose_command_uses_docker(tmp_path: Path) -> None:
    command = compose_command(tmp_path, tmp_path / "override.yaml", "start")

    assert command[:2] == ["docker", "compose"]
    assert not any("compose.extras.yaml" in part for part in command)
    assert not any(".env.override" in part for part in command)
    assert command[-5:] == [
        "up",
        "--no-build",
        "--force-recreate",
        "--remove-orphans",
        "--detach",
    ]


def test_docker_environment_pins_demo_version(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DOCKER_SOCK", str(tmp_path / "docker.sock"))
    environment = docker_environment(tmp_path / "flags.json")

    assert environment["DEMO_VERSION"] == "3.1.0"
    assert environment["IMAGE_VERSION"] == "3.1.0"
    assert environment["DOCKER_SOCK"] == str(tmp_path / "docker.sock")
    assert environment["TRACEMIND_FLAGD_CONFIG"] == str(
        (tmp_path / "flags.json").resolve()
    )


def test_docker_environment_finds_desktop_cli_on_stale_path(
    tmp_path: Path, monkeypatch
) -> None:
    docker_cli = tmp_path / ".docker" / "bin" / "docker"
    docker_cli.parent.mkdir(parents=True)
    docker_cli.touch()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    environment = docker_environment(tmp_path / "flags.json")

    assert environment["PATH"].split(os.pathsep)[0] == str(docker_cli.parent)


def test_prepare_flag_config_rejects_unknown_variant(tmp_path: Path) -> None:
    source_path = tmp_path / "src" / "flagd" / "demo.flagd.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        json.dumps(
            {
                "flags": {
                    "paymentFailure": {
                        "defaultVariant": "off",
                        "variants": {"off": 0},
                    }
                }
            }
        )
    )

    with pytest.raises(ValueError, match="unknown variant"):
        prepare_flag_config(
            tmp_path,
            tmp_path / "generated.json",
            feature_flag="paymentFailure",
            variant="100%",
        )
