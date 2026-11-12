"""Capture a real local two-release fault and run five primary DeepSeek cases.

Requires an existing local Node OpenTelemetry image and Demo collector/network.
Only the containers created here are removed. Build records and every primary
model attempt remain under the explicit output directory. No Demo flags change.
"""

import argparse
import asyncio
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import logging
from pathlib import Path
import shutil
import subprocess
from time import monotonic
from uuid import UUID

import httpx
from dotenv import dotenv_values

from tracemind.agent.storage.run_repository import RunRepository
from tracemind.agent.worker import RunWorker
from tracemind.api.app import create_app, get_request_settings
from tracemind.config import Settings, settings_files
from tracemind.evaluation.preflight import verify_captured_traces
from tracemind.mcp.client import HttpMcpClient
from tracemind.mcp.results import telemetry_reply
from tracemind.models.sources import SOURCE_TOOLS

ROOT = Path(__file__).resolve().parents[1]
BASE_IMAGE = (
    "ghcr.io/open-telemetry/demo@sha256:"
    "10d6bd20d8a00abe7705d553ab23a4f35e29b2816ab021049e63e89d07cc9192"
)


def save(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str)
    )


def command(*args, cwd=None, **kwargs):
    return subprocess.check_output(args, cwd=cwd, **kwargs).decode().strip()


def git(repo, *args):
    return command("git", "-C", str(repo), *args)


async def capture(args):
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    repo = folder / "fixture-repository"
    repo.mkdir()
    shutil.copyfile(ROOT / "evals/release_lab/app.js", repo / "app.js")
    (repo / "Dockerfile").write_text(
        f"FROM {args.base_image}\nCOPY app.js"
        ' /usr/src/app/release-fixture/app.js\nCMD ["--require",'
        ' "/otel-auto-instrumentation-nodejs/autoinstrumentation.js",'
        ' "release-fixture/app.js"]\n'
    )
    git(repo, "init", "-q")
    service = "tracemind-release-" + datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    records, running, releases = [], [], []
    cleanup = []
    try:
        for version in ("v1", "v2"):
            if version == "v2":
                app = repo / "app.js"
                app.write_text(
                    app.read_text().replace(
                        "input.currency || 'USD'", "input.currency"
                    )
                )
            git(repo, "add", "app.js", "Dockerfile")
            git(
                repo,
                "-c",
                "user.name=TraceMind Lab",
                "-c",
                "user.email=lab@example.invalid",
                "commit",
                "-qm",
                version,
            )
            commit = git(repo, "rev-parse", "HEAD")
            tree = git(repo, "rev-parse", "HEAD^{tree}")
            archive = subprocess.check_output(
                ["git", "-C", str(repo), "archive", commit]
            )
            archive_path = folder / (version + "-build-context.tar")
            archive_path.write_bytes(archive)
            image_file = folder / (version + "-image-id.txt")
            with (folder / (version + "-build.log")).open("wb") as log:
                subprocess.run(
                    [
                        args.docker,
                        "build",
                        "--pull=false",
                        "--network=none",
                        "--iidfile",
                        str(image_file),
                        "-",
                    ],
                    input=archive,
                    stdout=log,
                    stderr=log,
                    check=True,
                )
            image_id = image_file.read_text().strip()
            instance = service + "-" + version
            built_at = datetime.now(UTC).isoformat()
            container = command(
                args.docker,
                "run",
                "-d",
                "--name",
                instance,
                "--network",
                args.network,
                "--publish",
                "127.0.0.1::8080",
                "-e",
                "NODE_OPTIONS=",
                "-e",
                "OTEL_SERVICE_NAME=" + service,
                "-e",
                "OTEL_RESOURCE_ATTRIBUTES=service.version="
                + version
                + ",service.instance.id="
                + instance
                + ",deployment.environment.name=phase2-lab",
                "-e",
                "OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318",
                "-e",
                "OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf",
                "-e",
                "OTEL_TRACES_EXPORTER=otlp",
                "-e",
                "OTEL_METRICS_EXPORTER=none",
                "-e",
                "OTEL_LOGS_EXPORTER=none",
                "-e",
                "OTEL_TRACES_SAMPLER=always_on",
                "-e",
                "OTEL_NODE_ENABLED_INSTRUMENTATIONS=http",
                "-e",
                "OTEL_BSP_SCHEDULE_DELAY=100",
                image_id,
            )
            running.append(container)
            deployed = json.loads(
                command(
                    args.docker,
                    "inspect",
                    "--format",
                    '{"image":{{json .Image}},"container":{{json'
                    ' .Id}},"started_at":{{json .State.StartedAt}}}',
                    container,
                )
            )
            assert deployed["image"] == image_id
            port = command(args.docker, "port", container, "8080/tcp").rsplit(
                ":", 1
            )[1]
            record = dict(
                version=version,
                commit=commit,
                source_tree=tree,
                image_id=image_id,
                build_context_sha256=sha256(archive).hexdigest(),
                base_image=args.base_image,
                captured_at=built_at,
                runtime=deployed,
                port=int(port),
            )
            records.append(record)
            release = dict(
                id=version,
                service=service,
                repository_id="release-lab",
                commit=commit,
                version=version,
                scope={"environment": "phase2-lab"},
                deployed_at=deployed["started_at"],
                instances=[instance],
                containers=[container],
                image_id=image_id,
                previous_release_id="v1" if version == "v2" else None,
                provenance=(
                    "Local git archive -> docker build --network=none ->"
                    " inspected running image; capture.json"
                ),
                captured_at=datetime.now(UTC).isoformat(),
                build=dict(
                    record_id=version + "-build",
                    captured_at=built_at,
                    producer="release_source_lab.py",
                    source_tree=tree,
                    image_id=image_id,
                ),
            )
            releases.append(release)
        policy = dict(
            repositories=[
                dict(
                    id="release-lab",
                    path=str(repo),
                    service=service,
                    runtime_paths={"/usr/src/app/release-fixture": "."},
                )
            ],
            releases=releases,
        )
        save(folder / "catalog.json", policy)
        save(
            folder / "catalog-missing.json",
            {**policy, "releases": releases[:1]},
        )
        requests = []
        async with httpx.AsyncClient(timeout=10) as client:
            for record in records:
                url = f"http://127.0.0.1:{record['port']}"
                for attempt in range(40):
                    try:
                        if (
                            await client.get(url + "/health")
                        ).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    await asyncio.sleep(0.25)
                else:
                    raise RuntimeError("lab service did not become ready")
            start = datetime.now(UTC) - timedelta(seconds=1)
            for name, index, body, expected in [
                ("old-missing", 0, {}, 200),
                ("new-missing", 1, {}, 500),
                ("new-valid", 1, {"currency": "eur"}, 200),
                ("old-object", 0, {"currency": {}}, 500),
                ("new-object", 1, {"currency": {}}, 500),
            ]:
                record = records[index]
                response = await client.post(
                    f"http://127.0.0.1:{record['port']}/quote", json=body
                )
                observed = response.json()
                if response.status_code != expected or set(
                    observed["trace_id"]
                ) == {"0"}:
                    raise RuntimeError(
                        "lab request did not produce expected HTTP/trace"
                    )
                requests.append(
                    dict(
                        name=name,
                        version=record["version"],
                        input=body,
                        http_status=response.status_code,
                        **observed,
                    )
                )
            end = datetime.now(UTC) + timedelta(seconds=1)
        captured = {"traces": []}
        mcp = HttpMcpClient(args.mcp_url)
        try:
            await mcp.discover_tools()
            for request in requests:
                for attempt in range(40):
                    try:
                        reply = telemetry_reply(
                            await mcp.call(
                                "get_trace_detail",
                                {
                                    "trace_id": request["trace_id"],
                                    "limit": 100,
                                },
                            )
                        )
                        if reply.evidence:
                            break
                    except Exception:
                        if attempt == 39:
                            raise
                    await asyncio.sleep(0.5)
                else:
                    raise RuntimeError("trace not exported")
                target = next(
                    e
                    for e in reply.evidence
                    if e.attributes.get("span_id") == request["span_id"]
                )
                resource = target.attributes.get("resource", {})
                assert resource["service.version"] == request["version"]
                assert resource["deployment.environment.name"] == "phase2-lab"
                assert bool(target.attributes["is_error"]) == (
                    request["http_status"] == 500
                )
                captured["traces"].extend(
                    e.model_dump(mode="json") for e in reply.evidence
                )
            preflight = await verify_captured_traces(mcp, captured)
        finally:
            await mcp.aclose()
        capture_data = dict(
            service=service,
            start=start.isoformat(),
            end=end.isoformat(),
            builds=records,
            requests=requests,
        )
        save(folder / "capture.json", capture_data)
        save(folder / "telemetry.json", captured)
        save(folder / "capture-preflight.json", preflight)
        shutil.copyfile(__file__, folder / "campaign-script.py")
        print(
            json.dumps(
                {
                    "capture": str(folder),
                    "requests": len(requests),
                    "preflight": preflight,
                }
            ),
            flush=True,
        )
    finally:
        for container in reversed(running):
            logs = subprocess.run(
                [args.docker, "logs", container], capture_output=True
            )
            (folder / (container[:12] + "-container.log")).write_bytes(
                logs.stdout + logs.stderr
            )
            result = subprocess.run(
                [args.docker, "rm", "-f", container],
                capture_output=True,
                text=True,
            )
            cleanup.append(
                {"container": container, "removed": result.returncode == 0}
            )
        save(folder / "cleanup.json", cleanup)


def evaluation_settings(args) -> Settings:
    """Share API/worker configuration, with explicit lab-only overrides."""
    overrides = {}
    if args.mcp_url:
        overrides["telemetry_mcp_url"] = args.mcp_url
    if args.credential_file:
        # Preserve the existing key-only CLI option for older lab invocations.
        supplied = dotenv_values(args.credential_file)
        key = supplied.get("TRACEMIND_DEEPSEEK_API_KEY") or supplied.get(
            "OPS_PILOT_DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("authorized DeepSeek credential is missing")
        overrides["deepseek_api_key"] = key
    return Settings(_env_file=settings_files(args.env_file), **overrides)


async def evaluate(args, *, selected_cases=None, primary=True):
    profile = evaluation_settings(args)
    if profile.agent_mode != "single":
        raise ValueError(
            "source release acceptance requires single-agent mode")
    folder = args.output.resolve()
    capture_data = json.loads((folder / "capture.json").read_text())
    captured = json.loads((folder / "telemetry.json").read_text())
    if profile.llm_provider == "deepseek" and not profile.deepseek_api_key:
        raise RuntimeError("authorized DeepSeek credential is missing")
    requests = {r["name"]: r for r in capture_data["requests"]}
    service = capture_data["service"]
    canary_description = (
        f"POST /quote on {service}: the supplied old-version missing-currency"
        " request returned HTTP 200, the new-version missing-currency request"
        " returned HTTP 500, and the new-version EUR request returned HTTP"
        " 200. Identify the failure location, assess a possible release"
        " relationship, and propose a precise next check. These are"
        " individual requests, not population error rates."
    )
    cases = [
        (
            "exception-source",
            ["new-missing"],
            "catalog.json",
            (
                f"POST /quote on {service} returned HTTP 500 for an empty JSON"
                " object. Locate the error and explain the code mechanism if"
                " established, with a concrete next check."
            ),
        ),
        (
            "canary-source",
            ["old-missing", "new-missing", "new-valid"],
            "catalog.json",
            canary_description,
        ),
        (
            "canary-telemetry",
            ["old-missing", "new-missing", "new-valid"],
            None,
            canary_description,
        ),
        (
            "missing-mapping",
            ["new-missing"],
            "catalog-missing.json",
            (
                f"POST /quote on {service} returned HTTP 500 for an empty JSON"
                " object. Locate the error, assess release evidence and state"
                " any source mapping gaps."
            ),
        ),
        (
            "counterexample",
            ["old-object", "new-object"],
            "catalog.json",
            (
                f"POST /quote on {service} returned HTTP 500 on both supplied"
                " old and new version requests with currency set to an"
                " object. Investigate whether the new release explains this"
                " failure and give the next code inspection."
            ),
        ),
    ]
    if selected_cases is not None:
        cases = [case for case in cases if case[0] in selected_cases]
    save(
        folder / "protocol.json",
        {
            "cases": [c[0] for c in cases],
            "primary_attempts_only": primary,
            "model_budgets": profile.model_budgets,
            "provider": profile.llm_provider,
            "model": profile.llm_model,
            "verifier_model": profile.verifier_model,
            "fallback_provider": profile.agent_fallback_provider,
            "fallback_model": profile.agent_fallback_model,
            "knowledge_enabled": profile.agent_knowledge_enabled,
            "memory_enabled": profile.agent_memory_enabled,
            "investigator_reasoning": profile.deepseek_reasoning_effort,
            "verifier_reasoning": profile.verifier_reasoning_effort,
            "max_model_calls_per_case": profile.agent_max_model_calls,
            "max_tool_calls_per_case": profile.agent_max_tool_calls,
            "max_source_calls_per_case": 6,
            "idle_timeout_seconds": profile.agent_model_timeout_seconds,
            "overall_timeout_seconds": profile.agent_timeout_seconds,
            "verifier_enabled": profile.verifier_enabled,
            "revision_rounds": int(profile.verifier_enabled
                                   and profile.verifier_revise_on_rejection),
            "comparison": (
                "canary-source and canary-telemetry share"
                " input/window/backend; one attempt per condition; no accuracy"
                " or speed superiority inference"
            ),
            "ground_truth": (
                "v2 removes default for absent currency; object.toUpperCase"
                " fails in both versions and is not introduced by that diff"
            ),
            "git_head": git(ROOT, "rev-parse", "HEAD"),
            "working_diff_sha256": (
                sha256(
                    subprocess.check_output(["git", "diff"], cwd=ROOT)
                ).hexdigest()
            ),
        },
    )
    ledger = []
    for name, names, catalog, description in cases:
        case = folder / name
        case.mkdir(exist_ok=False)
        mcp = HttpMcpClient(args.mcp_url)
        try:
            proof = await verify_captured_traces(mcp, captured)
            save(case / "preflight.json", proof)
        finally:
            await mcp.aclose()
        settings = profile.model_copy(update={
            "agent_source_catalog": folder / catalog if catalog else None,
            "agent_allowed_tools": (
                (*tuple(t for t in profile.agent_allowed_tools
                        if t not in SOURCE_TOOLS), *SOURCE_TOOLS)
                if catalog else tuple(t for t in profile.agent_allowed_tools
                                      if t not in SOURCE_TOOLS)
            ),
            "agent_trace_directory": case / "events",
            "agent_run_database": case / "runs.sqlite3",
            "agent_checkpoint_database": case / "graph.sqlite3",
        })
        save(
            case / "settings.json",
            dict(
                limits=settings.run_limits.model_dump(mode="json"),
                verification=(settings.verification_policy.model_dump(
                    mode="json") if settings.verification_policy else None),
                allowed_tools=settings.agent_allowed_tools,
                source_snapshot_id=(
                    settings.source_policy.snapshot_id if catalog else None
                ),
            ),
        )
        payload = {
            "incident": {
                "title": "Quote request failure",
                "description": description,
                "affected_services": [service],
            },
            "investigation_start": capture_data["start"],
            "investigation_end": capture_data["end"],
            "initial_context": {
                "scope": {"environment": "phase2-lab"},
                "traces": [
                    {
                        "trace_id": requests[n]["trace_id"],
                        "span_id": requests[n]["span_id"],
                        "service": service,
                    }
                    for n in names
                ],
            },
        }
        save(case / "input.json", payload)
        app = create_app()
        app.dependency_overrides[get_request_settings] = lambda: settings
        repository = RunRepository(settings.agent_run_database)
        started = monotonic()
        first = None
        run_id = None
        print(json.dumps({"case": name, "status": "starting"}), flush=True)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as api:
                response = await api.post(
                    "/api/v1/runs",
                    json=payload,
                    headers={"Idempotency-Key": name},
                )
                response.raise_for_status()
                save(case / "created.json", response.json())
                run_id = UUID(response.json()["id"])
                route = f"/api/v1/runs/{run_id}"
                task = asyncio.create_task(
                    RunWorker(settings, repository=repository).run_once()
                )
                while not task.done():
                    response = await api.get(route + "/triage")
                    if response.status_code == 200 and first is None:
                        first = monotonic() - started
                        save(
                            case / "early-triage.json",
                            {
                                "elapsed_seconds": first,
                                "body": response.json(),
                            },
                        )
                        print(
                            json.dumps(
                                {
                                    "case": name,
                                    "first_triage_seconds": round(first, 2),
                                }
                            ),
                            flush=True,
                        )
                    await asyncio.sleep(0.25)
                await task
                for endpoint in ["triage", "result"]:
                    response = await api.get(route + "/" + endpoint)
                    save(
                        case / (endpoint + ".json"),
                        {
                            "http_status": response.status_code,
                            "body": response.json(),
                        },
                    )
        finally:
            if run_id:
                stored = repository.get(run_id)
                state = stored.state
                save(case / "state.json", state.model_dump(mode="json"))
                save(
                    case / "evidence.json",
                    [e.model_dump(mode="json") for e in stored.evidence],
                )
                row = dict(
                    case=name,
                    run_id=str(run_id),
                    status=state.status.value,
                    stop_reason=state.stop_reason,
                    elapsed_seconds=monotonic() - started,
                    first_triage_seconds=first,
                    model_calls=state.model_calls,
                    tool_calls=state.tool_calls,
                    source_records=len(state.sources),
                    reviews=[{
                        "round": index,
                        "input_sha256": record.input_sha256,
                        "status": record.status,
                        "attempts": record.attempts,
                        "verdict": (record.review.verdict
                                    if record.review else None),
                    } for index, record in enumerate([
                        *state.verification_history,
                        *([state.verification] if state.verification else []),
                    ])],
                    review_status=(
                        state.verification.status
                        if state.verification
                        else None
                    ),
                    review_error=(
                        state.verification.error_code
                        if state.verification
                        else None
                    ),
                )
                ledger.append(row)
                save(folder / "ledger.json", ledger)
                print(json.dumps(row, default=str), flush=True)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--docker",
        default=shutil.which("docker") or "docker",
        help="Docker executable; pass an absolute path if not on PATH",
    )
    parser.add_argument("--network", default="opentelemetry-demo")
    parser.add_argument("--base-image", default=BASE_IMAGE)
    parser.add_argument("--mcp-url",
                        help="override the configured MCP endpoint")
    parser.add_argument("--env-file", type=Path, action="append",
                        help="overlay on the application .env")
    parser.add_argument("--credential-file", type=Path,
                        help="legacy key-only override; otherwise use .env")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--capture-only", action="store_true")
    mode.add_argument("--run-captured", action="store_true")
    args = parser.parse_args()
    args.mcp_url = str(evaluation_settings(args).telemetry_mcp_url)
    logging.basicConfig(level=logging.WARNING)
    if not args.run_captured:
        await capture(args)
    if not args.capture_only:
        await evaluate(args)


if __name__ == "__main__":
    asyncio.run(main())
