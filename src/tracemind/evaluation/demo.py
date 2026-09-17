"""Pinned OpenTelemetry Demo configuration and Docker Compose helpers."""

import json
import os
import shutil
import subprocess
from pathlib import Path


DEMO_COMMIT = "dedc0178918e260823323b8d95005a8cb924b007"
SCENARIO_NAMES = (
    "payment_failure", "cart_failure", "shipping_slowdown",
    "product_catalog_lock_contention", "payment_unreachable",
)


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


def activate_flag_config(
    prepared_path: Path, active_path: Path, backup_path: Path,
) -> Path:
    """Save the previous configuration before updating the mounted file.

    Write in place so an existing Docker file bind keeps the same inode.
    Callers restart flagd after the complete write, before using new data.
    """
    content = prepared_path.read_bytes()
    if active_path.exists():
        backup_path.write_bytes(active_path.read_bytes())
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_bytes(content)
    return active_path.resolve()


def restore_flag_config(active_path: Path, backup_path: Path) -> Path:
    """Restore the last saved configuration without replacing its backup."""
    content = backup_path.read_bytes()
    active_path.write_bytes(content)
    return active_path.resolve()


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
    """Build, but do not execute, a Docker Compose command for the
    scenario.

    Support start, switch, restore, stop, and status. The stop
    command includes volume removal, while start recreates containers
    without building.
    """
    files = [
        demo_path / "compose.yaml",
        demo_path / "compose.observability.yaml",
        override_path,
    ]
    command = [
        "docker",
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
    elif action in {"switch", "restore"}:
        command.extend(("up", "--no-build", "--force-recreate", "--detach",
                        "--no-deps", "flagd"))
    elif action == "stop":
        command.extend(("down", "--remove-orphans", "--volumes"))
    elif action == "status":
        command.append("ps")
    else:
        raise ValueError(f"unsupported compose action: {action}")
    return command


def docker_environment(flag_config: Path) -> dict[str, str]:
    """Copy the process environment and add pinned Demo settings."""
    environment = os.environ.copy()
    desktop_cli = Path.home() / ".docker" / "bin" / "docker"
    if shutil.which("docker", path=environment.get("PATH")) is None and (
        desktop_cli.is_file()
    ):
        environment["PATH"] = (
            f"{desktop_cli.parent}{os.pathsep}{environment.get('PATH', '')}"
        )
    desktop_socket = Path.home() / ".docker" / "run" / "docker.sock"
    environment.update(
        {
            "DOCKER_SOCK": environment.get("DOCKER_SOCK") or str(
                desktop_socket if desktop_socket.is_socket()
                else Path("/var/run/docker.sock")
            ),
            "DEMO_VERSION": "3.1.0",
            "IMAGE_VERSION": "3.1.0",
            "TRACEMIND_FLAGD_CONFIG": str(flag_config.resolve()),
            "TRACEMIND_JAEGER_CONFIG": str(
                Path(__file__).resolve().parents[3]
                / "deploy" / "jaeger-persistent.yaml"
            ),
        }
    )
    return environment
