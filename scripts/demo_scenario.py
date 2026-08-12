import argparse
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from tracemind.evaluation.demo import (
    compose_command,
    podman_environment,
    prepare_flag_config,
    verify_demo_commit,
)
from tracemind.evaluation.models import EvaluationCase


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_PATH = PROJECT_ROOT.parent / "opentelemetry-demo"
CASE_PATH = PROJECT_ROOT / "evals" / "cases" / "payment_failure.json"
OVERRIDE_PATH = PROJECT_ROOT / "evals" / "compose.payment-failure.yaml"
GENERATED_FLAG_PATH = (
    PROJECT_ROOT / "artifacts" / "otel-demo" / "payment-failure.flagd.json"
)


def wait_for_backends(timeout_seconds: int = 300) -> None:
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
    parser = argparse.ArgumentParser(
        description="Manage the payment failure scenario"
    )
    parser.add_argument(
        "action", choices=("prepare", "start", "status", "stop")
    )
    parser.add_argument("--demo-path", type=Path, default=DEFAULT_DEMO_PATH)
    args = parser.parse_args()

    case = EvaluationCase.load(CASE_PATH)
    verify_demo_commit(args.demo_path)
    flag_config = prepare_flag_config(
        args.demo_path,
        GENERATED_FLAG_PATH,
        feature_flag=case.fault.feature_flag,
        variant=case.fault.variant,
    )

    if args.action == "prepare":
        print(f"Prepared {flag_config}")
        return

    if shutil.which("podman") is None:
        raise SystemExit(
            "Podman is not installed. Run: sudo apt-get update && "
            "sudo apt-get install -y podman podman-compose"
        )

    subprocess.run(
        compose_command(args.demo_path, OVERRIDE_PATH, args.action),
        cwd=args.demo_path,
        env=podman_environment(flag_config),
        check=True,
    )
    if args.action == "start":
        wait_for_backends()
        print("Prometheus, Jaeger, and OpenSearch are ready")


if __name__ == "__main__":
    main()
