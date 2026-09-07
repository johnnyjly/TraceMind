"""Run agent, saved-diagnosis, verifier or RAG experiments in Langfuse."""

import argparse
from datetime import UTC, datetime, timedelta
import json
import logging
from math import isfinite
from pathlib import Path

from tracemind.config import Settings
from tracemind.evaluation.budget import ModelCallBudget
from tracemind.evaluation.models import EvaluationCase
from tracemind.evaluation.experiments import (
    accepted,
    run_experiment,
    verify_publication,
)
from tracemind.models.memory import MemorySnapshot
from tracemind.observability.langfuse import client_for


def configure_logging(*, verbose=False):
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s "
            "[run=%(run_id)s parent=%(parent_run_id)s] %(message)s",
            defaults={"run_id": "-", "parent_run_id": "-"},
        )
    )
    logging.basicConfig(level=logging.WARNING, handlers=[handler])
    logging.getLogger("tracemind").setLevel(logging.INFO)
    logging.getLogger("tracemind.agent.runtime.harness").setLevel(
        logging.DEBUG if verbose else logging.INFO
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--data", type=Path, help="JSON list of Langfuse experiment items"
    )
    source.add_argument(
        "--case", type=Path, help="Demo incident fixture; labels stay held out"
    )
    source.add_argument(
        "--dataset", help="Dataset name hosted in local Langfuse"
    )
    parser.add_argument("--lookback-minutes", type=float, default=10)
    parser.add_argument("--dataset-version", type=datetime.fromisoformat)
    parser.add_argument(
        "--task", choices=["agent", "saved", "retrieval", "verification"],
        required=True
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--env-file", type=Path, action="append", default=[])
    parser.add_argument("--judge", action="store_true")
    parser.add_argument(
        "--max-model-calls",
        type=int,
        required=True,
        help="Shared investigator/verifier/fallback/retry/RAG/judge cap",
    )
    parser.add_argument(
        "--min-score", action="append", default=[], metavar="NAME=MIN"
    )
    parser.add_argument("--memory-snapshot", type=Path)
    parser.add_argument("--compare-memory", action="store_true")
    parser.add_argument("--publish-timeout", type=float, default=30)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if (
        args.max_model_calls < 1
        or not isfinite(args.publish_timeout)
        or args.publish_timeout <= 0
    ):
        parser.error("call budget and publish timeout must be positive")
    if args.case and (
        args.task != "agent"
        or not isfinite(args.lookback_minutes)
        or args.lookback_minutes <= 0
    ):
        parser.error(
            "--case requires an agent task and positive finite lookback"
        )
    if args.dataset_version and not args.dataset:
        parser.error("--dataset-version requires --dataset")
    if args.compare_memory and not args.memory_snapshot:
        parser.error("--compare-memory requires --memory-snapshot")
    if args.memory_snapshot and args.task != "agent":
        parser.error("memory snapshots apply only to agent tasks")
    try:
        thresholds = dict(
            (name, float(value))
            for name, value in (item.rsplit("=", 1) for item in args.min_score)
        )
    except (ValueError, TypeError):
        parser.error("--min-score must be NAME=MIN")
    if any(
        not name or not isfinite(value) for name, value in thresholds.items()
    ):
        parser.error("score thresholds must have names and finite values")
    settings = Settings(_env_file=args.env_file or Path(".env"))
    client = client_for(settings, required=True)
    if not client.auth_check():
        raise SystemExit(
            "Langfuse authentication failed; no model calls admitted"
        )
    if args.case:
        case = EvaluationCase.load(args.case)
        end = datetime.now(UTC)
        data = [
            {
                "input": {
                    "incident": case.incident.model_dump(mode="json"),
                    "start": (
                        end - timedelta(minutes=args.lookback_minutes)
                    ).isoformat(),
                    "end": end.isoformat(),
                },
                "expected_output": case.ground_truth.model_dump(mode="json"),
            }
        ]
        settings = settings.model_copy(
            update={
                "agent_max_tool_calls": min(
                    settings.agent_max_tool_calls, case.max_tool_calls
                )
            }
        )
    else:
        data = (
            json.loads(args.data.read_text())
            if args.data
            else client.get_dataset(
                args.dataset, version=args.dataset_version
            ).items
        )
    if not isinstance(data, list) or not data:
        parser.error("dataset must contain at least one item")
    snapshot = (
        MemorySnapshot.model_validate_json(args.memory_snapshot.read_text())
        if args.memory_snapshot
        else None
    )
    variants = [None, snapshot] if args.compare_memory else [snapshot]
    budget = ModelCallBudget(args.max_model_calls)
    configure_logging(verbose=args.verbose)
    passed = True
    try:
        for variant in variants:
            from tracemind.rag.config import RagSettings

            result = run_experiment(
                settings,
                client,
                data,
                kind=args.task,
                name=args.name,
                budget=budget,
                judge=args.judge,
                thresholds=thresholds,
                memory_snapshot=variant,
                run_name=(
                    f"{args.run_name}--memory-"
                    f"{variant.fingerprint[:12] if variant else 'off'}"
                    if args.run_name and args.compare_memory
                    else args.run_name
                ),
                rag_settings=RagSettings(
                    _env_file=args.env_file or Path(".env")
                ),
            )
            print(result.format())
            print("Experiment:", result.experiment_id)
            published = verify_publication(
                client, result, timeout_seconds=args.publish_timeout
            )
            print("Scores and traces published:", published)
            passed = passed and accepted(result) and published
    finally:
        client.flush()
        print(f"Model calls admitted: {budget.calls}/{budget.limit}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
