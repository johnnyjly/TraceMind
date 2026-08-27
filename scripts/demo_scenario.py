"""Prepare, activate and restore isolated OpenTelemetry Demo fault cases."""

import argparse
import logging
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from tracemind.evaluation.demo import (
    SCENARIO_NAMES,
    activate_flag_config,
    compose_command,
    docker_environment,
    prepare_flag_config,
    restore_flag_config,
    verify_demo_commit,
)
from tracemind.evaluation.models import EvaluationCase


logger = logging.getLogger("tracemind.evaluation.demo_cli")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_PATH = PROJECT_ROOT.parent / "opentelemetry-demo"
CASE_PATH = PROJECT_ROOT / "evals" / "cases" / "payment_failure.json"
OVERRIDE_PATH = PROJECT_ROOT / "evals" / "compose.payment-failure.yaml"
GENERATED_FLAG_PATH = (
    PROJECT_ROOT / "artifacts" / "otel-demo" / "payment-failure.flagd.json"
)
BACKUP_FLAG_PATH = GENERATED_FLAG_PATH.with_name("previous.flagd.json")


def wait_for_backends(timeout_seconds: int = 300) -> None:
    """Wait for the existing telemetry backends after starting the Demo."""
    endpoints = {
        "Prometheus": "http://localhost:9090/-/ready",
        "Jaeger": "http://localhost:16686/jaeger/ui/api/v3/services",
        "OpenSearch": "http://localhost:9200/_cluster/health",
    }
    deadline = time.monotonic() + timeout_seconds
    pending = dict(endpoints)
    while pending and time.monotonic() < deadline:
        for name, url in list(pending.items()):
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status < 400:
                        pending.pop(name)
            except (urllib.error.URLError, TimeoutError):
                pass
        if pending:
            time.sleep(2)
    if pending:
        names = ", ".join(pending)
        raise RuntimeError(
            f"timed out waiting for telemetry backends: {names}"
        )


def main() -> None:
    """Select a case while keeping status and stop free of file mutations."""
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "action", choices=("prepare", "start", "switch", "restore",
                           "status", "stop")
    )
    parser.add_argument("--scenario", choices=SCENARIO_NAMES,
                        default="payment_failure")
    parser.add_argument("--demo-path", type=Path, default=DEFAULT_DEMO_PATH)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    verify_demo_commit(args.demo_path)
    environment = docker_environment(GENERATED_FLAG_PATH)

    if (args.action != "prepare" and
            shutil.which("docker", path=environment.get("PATH")) is None):
        raise SystemExit(
            "Docker CLI is unavailable. Install and start Docker Desktop, "
            "then check that docker is on PATH."
        )

    if args.action in {"prepare", "start", "switch"}:
        if args.action == "switch" and not GENERATED_FLAG_PATH.exists():
            parser.error("switch requires an existing Demo; use start first")
        case = EvaluationCase.load(
            CASE_PATH.with_name(f"{args.scenario}.json"))
        prepared = prepare_flag_config(
            args.demo_path,
            GENERATED_FLAG_PATH.with_name(f"{args.scenario}.flagd.json"),
            feature_flag=case.fault.feature_flag,
            variant=case.fault.variant,
        )
        if args.action == "prepare":
            logger.info("Prepared %s (not activated)", prepared)
            return
        activate_flag_config(prepared, GENERATED_FLAG_PATH, BACKUP_FLAG_PATH)
    elif args.action == "restore":
        if not BACKUP_FLAG_PATH.exists():
            parser.error("no previous flag configuration to restore")
        restore_flag_config(GENERATED_FLAG_PATH, BACKUP_FLAG_PATH)

    subprocess.run(
        compose_command(args.demo_path, OVERRIDE_PATH, args.action),
        cwd=args.demo_path,
        env=environment,
        check=True,
    )
    if args.action == "start":
        wait_for_backends()
        logger.info("Prometheus, Jaeger, and OpenSearch are ready")
    if args.action in {"start", "switch", "restore"}:
        logger.info("Flags activated; collect fresh telemetry "
                    "before diagnosis")
        logger.info("restore returns to the configuration before the last "
                    "start/switch; retained telemetry is not deleted")


if __name__ == "__main__":
    main()
