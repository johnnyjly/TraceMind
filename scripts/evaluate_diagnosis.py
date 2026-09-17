"""Save a bounded live trial, including failures, usage and source identity."""

import argparse
import asyncio
from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
from threading import Lock
from time import monotonic

from tracemind.config import Settings
from tracemind.evaluation.preflight import verify_live_window
from tracemind.evaluation.budget import ModelCallBudget
from tracemind.evaluation.experiments import (
    accepted,
    run_experiment,
    verify_publication,
)
from tracemind.observability.langfuse import client_for

ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    return {
        str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
        for folder in ["src", "services", "evals/cases", "scripts"]
        for p in sorted((ROOT / folder).rglob("*"))
        if p.is_file() and p.suffix in {".py", ".json"}
    }


def load_profile(path):
    """Load the repository's literal, credential-free experiment profile."""
    values = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        if not key.startswith("TRACEMIND_"):
            raise ValueError("profile contains a non-application setting")
        values[key] = value.strip().strip("'\"")
    os.environ.update(values)
    return values


class UsageBudget(ModelCallBudget):
    def __init__(self, limit, ledger, usage):
        super().__init__(limit, ledger_path=ledger)
        self.usage = usage
        self.started = {}
        self.usage_lock = Lock()

    def record(self, **event):
        with self.usage_lock, self.usage.open("a") as stream:
            stream.write(
                json.dumps({"at": datetime.now(UTC).isoformat(), **event})
                + "\n"
            )

    def on_chat_model_start(self, serialized, messages, **kwargs):
        super().on_chat_model_start(serialized, messages, **kwargs)
        run_id = str(kwargs["run_id"])
        self.started[run_id] = monotonic()
        params = kwargs.get("invocation_params") or {}
        self.record(
            event="admitted",
            run_id=run_id,
            model=params.get("model", params.get("model_name")),
        )

    def on_llm_end(self, response, **kwargs):
        run_id = str(kwargs["run_id"])
        self.record(
            event="completed",
            run_id=run_id,
            elapsed_ms=(monotonic() - self.started[run_id]) * 1000,
            usage=[
                getattr(g.message, "usage_metadata", None)
                for batch in response.generations
                for g in batch
                if hasattr(g, "message")
            ],
        )

    def on_llm_error(self, error, **kwargs):
        run_id = str(kwargs["run_id"])
        self.record(
            event="error",
            run_id=run_id,
            error_type=type(error).__name__,
            elapsed_ms=(monotonic() - self.started.get(run_id, monotonic()))
            * 1000,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--env-file", type=Path, action="append", required=True)
    p.add_argument("--mode", choices=["single", "multi"], required=True)
    p.add_argument("--max-model-calls", type=int, default=40)
    p.add_argument("--without-knowledge", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("tracemind").setLevel(logging.INFO)
    preflight = json.loads((args.data.parent / "preflight.json").read_text())
    if not preflight.get("reproduced"):
        raise ValueError("fault window did not pass reproduction preflight")
    if (
        preflight.get("input_sha256")
        != sha256(args.data.read_bytes()).hexdigest()
    ):
        raise ValueError("evaluation input differs from validated window")
    if args.output.exists():
        raise ValueError(
            "trial output already exists; choose a new trial name"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile = load_profile(args.profile)
    os.environ.update(
        NO_PROXY="localhost,127.0.0.1,::1", no_proxy="localhost,127.0.0.1,::1"
    )
    settings = Settings(
        _env_file=args.env_file,
        agent_mode=args.mode,
        langfuse_enabled=True,
        agent_trace_directory=args.output.parent / "runs",
        agent_run_database=args.output.parent / "runs.sqlite3",
        agent_checkpoint_database=args.output.parent / "checkpoints.sqlite3",
    )
    if args.without_knowledge:
        settings = settings.model_copy(
            update={
                "agent_knowledge_enabled": False,
                "agent_allowed_tools": tuple(
                    t
                    for t in settings.agent_allowed_tools
                    if t not in {"search_knowledge", "read_knowledge"}
                ),
            }
        )
    if preflight.get("mcp_url") and preflight["mcp_url"] != str(
        settings.telemetry_mcp_url
    ):
        raise ValueError("preflight and investigator MCP endpoints differ")
    live_window = asyncio.run(verify_live_window(
        str(settings.telemetry_mcp_url),
        json.loads((args.data.parent / "telemetry.json").read_text()),
    ))
    budget = UsageBudget(
        args.max_model_calls,
        args.output.with_suffix(".ledger.sqlite3"),
        args.output.with_suffix(".usage.jsonl"),
    )
    client = client_for(settings, required=True)
    hashes = source_hashes()
    trial_name = (
        f"{args.data.parent.parent.name}-{args.mode}-{args.output.stem}"
    )
    record = {
        "name": trial_name,
        "mode": args.mode,
        "started_at": datetime.now(UTC).isoformat(),
        "profile": profile,
        "knowledge": settings.agent_knowledge_enabled,
        "source_hashes": hashes,
        "profile_sha256": sha256(args.profile.read_bytes()).hexdigest(),
        "mcp_url": str(settings.telemetry_mcp_url),
        "live_window": live_window,
        "data_sha256": sha256(args.data.read_bytes()).hexdigest(),
        "limits": settings.run_limits.model_dump(mode="json"),
    }
    try:
        if not client.auth_check():
            raise RuntimeError(
                "Langfuse authentication failed before model admission"
            )
        result = run_experiment(
            settings,
            client,
            json.loads(args.data.read_text()),
            kind="agent",
            name=trial_name,
            budget=budget,
            judge=True,
            thresholds={"required_source_citation": 1.0},
        )

        def score(value):
            return {
                k: getattr(value, k, None)
                for k in ["name", "value", "comment", "metadata"]
            }

        record.update(
            accepted=accepted(result),
            experiment_id=result.experiment_id,
            items=[
                {
                    "trace_id": i.trace_id,
                    "output": i.output,
                    "evaluations": [score(s) for s in i.evaluations],
                }
                for i in result.item_results
            ],
            run_evaluations=[score(s) for s in result.run_evaluations],
        )
        args.output.write_text(json.dumps(record, indent=2))
        record["publication_verified"] = verify_publication(
            client, result, timeout_seconds=60
        )
    except Exception as exc:
        record.setdefault("accepted", False)
        record["exception_type"] = type(exc).__name__
    finally:
        record.update(
            finished_at=datetime.now(UTC).isoformat(),
            calls_admitted=budget.calls,
            source_unchanged=hashes == source_hashes(),
        )
        args.output.write_text(json.dumps(record, indent=2))
        print(
            json.dumps(
                {
                    k: record.get(k)
                    for k in [
                        "name",
                        "mode",
                        "accepted",
                        "calls_admitted",
                        "publication_verified",
                        "exception_type",
                    ]
                }
            ),
            flush=True,
        )
        client.shutdown()


if __name__ == "__main__":
    main()
