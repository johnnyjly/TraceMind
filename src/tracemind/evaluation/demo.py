"""Pinned OpenTelemetry Demo configuration and Podman command helpers."""

import json
import os
import subprocess
from pathlib import Path


DEMO_COMMIT = "dedc0178918e260823323b8d95005a8cb924b007"


def prepare_flag_config(
    demo_path: Path,
    output_path: Path,
    *,
    feature_flag: str,
    variant: str,
) -> Path:
    """Write a copy of Demo flag configuration with one default changed.

    Reject unknown flags or variants, create the output directory, and
    return the resolved output path. Other flags retain their source
    configuration.
    """
    source = demo_path / "src" / "flagd" / "demo.flagd.json"
    document = json.loads(source.read_text())
    flag = document["flags"].get(feature_flag)
    if not isinstance(flag, dict):
        raise ValueError(f"unknown Demo feature flag: {feature_flag}")
    variants = flag.get("variants")
    if not isinstance(variants, dict) or variant not in variants:
        raise ValueError(f"unknown variant {variant!r} for {feature_flag}")

    flag["defaultVariant"] = variant
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2) + "\n")
    return output_path.resolve()


def verify_demo_commit(demo_path: Path) -> None:
    """Reject a Demo checkout whose HEAD differs from the pinned commit."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=demo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip() != DEMO_COMMIT:
        raise RuntimeError(
            f"OpenTelemetry Demo must be pinned to {DEMO_COMMIT}"
        )


def compose_command(
    demo_path: Path,
    override_path: Path,
    action: str,
) -> list[str]:
    """Build, but do not execute, a Podman Compose command for the
    scenario.

    Support start, stop, and status; reject other actions. The stop
    command includes volume removal, while start recreates containers
    without building.
    """
    files = [
        demo_path / "compose.yaml",
        demo_path / "compose.observability.yaml",
        override_path,
    ]
    command = [
        "podman",
        "compose",
        "--env-file",
        str(demo_path / ".env"),
    ]
    for path in files:
        command.extend(("-f", str(path)))

    if action == "start":
        command.extend(
            (
                "up",
                "--no-build",
                "--force-recreate",
                "--remove-orphans",
                "--detach",
            )
        )
    elif action == "stop":
        command.extend(("down", "--remove-orphans", "--volumes"))
    elif action == "status":
        command.append("ps")
    else:
        raise ValueError(f"unsupported compose action: {action}")
    return command


def podman_environment(flag_config: Path) -> dict[str, str]:
    """Copy the process environment and add pinned Demo/Podman settings."""
    environment = os.environ.copy()
    runtime_dir = environment.get(
        "XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"
    )
    environment.update(
        {
            "DOCKER_SOCK": f"{runtime_dir}/podman/podman.sock",
            "DEMO_VERSION": "3.1.0",
            "IMAGE_VERSION": "3.1.0",
            "TRACEMIND_FLAGD_CONFIG": str(flag_config.resolve()),
            "PODMAN_COMPOSE_PROVIDER": "podman-compose",
            "POSTGRES_IMAGE": "docker.io/library/postgres:18.4",
            "GRAFANA_IMAGE": "docker.io/grafana/grafana:13.1.0",
        }
    )
    return environment
