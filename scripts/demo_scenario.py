"""Prepare, activate and restore isolated OpenTelemetry Demo fault cases."""

import argparse
import json
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
DEFAULT_DEMO_PATH = PROJECT_ROOT / ".local" / "opentelemetry-demo"
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
                        if name == "OpenSearch" and json.load(response).get(
                                "status") not in {"green", "yellow"}:
                            continue
                        pending.pop(name)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                pass
        if pending:
            time.sleep(2)
    if pending:
        names = ", ".join(pending)
        raise RuntimeError(
            f"timed out waiting for telemetry backends: {names}"
        )


def reset_demo(
    demo_path: Path, environment: dict[str, str], *, dry_run: bool = False,
) -> None:
    """Replace only this Demo's containers and data, keeping active flags."""
    command = compose_command(demo_path, OVERRIDE_PATH, "status")[:-1]
    command[2:2] = ["--project-name", "opentelemetry-demo"]

    def run(arguments: list[str]) -> str:
        return subprocess.run(
            arguments, cwd=demo_path, env=environment, check=True,
            capture_output=True, text=True,
        ).stdout

    config = json.loads(run(command + ["config", "--format", "json"]))
    # Validate local images and configuration before deleting a working stack.
    images = sorted({service["image"]
                     for service in config["services"].values()})
    run(["docker", "image", "inspect", *images])
    container_ids = run(["docker", "ps", "--all", "--quiet"]).split()
    containers = (json.loads(run(["docker", "inspect", *container_ids]))
                  if container_ids else [])
    targets = []
    for container in containers:
        labels = container["Config"].get("Labels") or {}
        if labels.get("com.docker.compose.project") != config["name"]:
            continue
        working_dir = labels.get("com.docker.compose.project.working_dir")
        if not working_dir or Path(working_dir).resolve() != demo_path.resolve():
            raise RuntimeError("Demo project belongs to another checkout")
        targets.append(container)

    existing_volumes = set(run(["docker", "volume", "ls", "--quiet"]).split())
    volumes = {mount["Name"] for container in targets
               for mount in container["Mounts"] if mount["Type"] == "volume"}
    declared = config.get("volumes", {}).values()
    volumes.update(volume["name"] for volume in declared
                   if volume["name"] in existing_volumes)
    target_ids = {container["Id"] for container in targets}
    for container in containers:
        if container["Id"] in target_ids:
            continue
        shared = volumes.intersection(
            mount["Name"] for mount in container["Mounts"]
            if mount["Type"] == "volume")
        if shared:
            raise RuntimeError(
                f"refusing to delete volumes used outside the Demo: {shared}")

    logger.warning("Demo reset will remove %d containers and volumes: %s",
                   len(targets), ", ".join(sorted(volumes)) or "(none)")
    logger.info("Active flags are preserved: %s",
                environment["TRACEMIND_FLAGD_CONFIG"])
    if dry_run:
        return
    if targets:
        run(["docker", "rm", "--force", "--volumes", *sorted(target_ids)])
    # Container removal already deletes anonymous volumes. External Jaeger
    # storage (including a mounted legacy volume) needs explicit deletion.
    remaining = set(run(["docker", "volume", "ls", "--quiet"]).split())
    for name in sorted(volumes & remaining):
        run(["docker", "volume", "rm", name])
    for volume in declared:
        if volume.get("external"):
            run(["docker", "volume", "create", volume["name"]])
    start = compose_command(demo_path, OVERRIDE_PATH, "start")
    start[2:2] = ["--project-name", "opentelemetry-demo"]
    subprocess.run(
        start + ["--pull", "never", "--wait", "--wait-timeout", "300"],
        cwd=demo_path, env=environment, check=True,
    )
    wait_for_backends()
    logger.info("Demo recreated; capture fresh incident windows before evaluation")


def main() -> None:
    """Select a case while keeping status and stop free of file mutations."""
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "action", choices=("prepare", "start", "switch", "restore",
                           "status", "stop", "reset")
    )
    parser.add_argument("--scenario", choices=SCENARIO_NAMES,
                        default="payment_failure")
    parser.add_argument("--demo-path", type=Path, default=DEFAULT_DEMO_PATH)
    parser.add_argument("--purge-data", action="store_true",
                        help="confirm deletion of Demo data for reset")
    parser.add_argument("--dry-run", action="store_true",
                        help="show reset targets without changing Docker")
    parser.add_argument("--flag-config", type=Path,
                        help="existing active flag file to preserve during reset")
    args = parser.parse_args()
    if args.action == "reset":
        if not (args.purge_data or args.dry_run):
            parser.error("reset requires --purge-data (or --dry-run to inspect)")
        active_flags = args.flag_config or GENERATED_FLAG_PATH
        if not active_flags.is_file():
            parser.error("reset needs an existing active --flag-config file")
    else:
        if args.purge_data or args.dry_run or args.flag_config:
            parser.error("--purge-data, --dry-run and --flag-config require reset")
        active_flags = GENERATED_FLAG_PATH
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    verify_demo_commit(args.demo_path)
    environment = docker_environment(active_flags)

    if (args.action != "prepare" and
            shutil.which("docker", path=environment.get("PATH")) is None):
        raise SystemExit(
            "Docker CLI is unavailable. Install and start Docker Desktop, "
            "then check that docker is on PATH."
        )

    if args.action == "reset":
        reset_demo(args.demo_path, environment, dry_run=args.dry_run)
        return

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

    if args.action == "start":
        subprocess.run(
            ["docker", "volume", "create", "tracemind-jaeger-data"],
            env=environment, check=True, capture_output=True,
        )
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
