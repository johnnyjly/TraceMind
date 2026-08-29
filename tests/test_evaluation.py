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
from tracemind.evaluation.evaluator import evaluate_diagnosis
from tracemind.evaluation.models import EvaluationCase
from tracemind.models import (
    Diagnosis,
    DiagnosisStatus,
    Evidence,
    EvidenceSource,
)
from tracemind.models.result import DiagnosisRun


CASE_PATH = Path("evals/cases/payment_failure.json")


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

    result = evaluate_diagnosis(case, run, elapsed_ms=250)

    assert result.passed is True
    assert result.required_evidence_match is True
    assert result.tool_call_count == 2


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

    result = evaluate_diagnosis(case, run, elapsed_ms=100)

    assert result.passed is False
    assert result.root_cause_service_match is False
    assert result.required_evidence_match is False


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
