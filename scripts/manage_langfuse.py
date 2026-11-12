"""Set up the local UI, publish run metrics, and export human-labelled evals."""

import argparse
import json
from pathlib import Path

from tracemind.config import Settings
from tracemind.models import Evidence
from tracemind.models.run import RunState
from tracemind.observability.langfuse import client_for
from tracemind.observability.metrics import run_metrics, publish_run_metrics
from tracemind.observability.project import bootstrap
from tracemind.evaluation.annotations import (
    enqueue_reviews, collect_annotations, annotation_dataset,
)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    setup = subs.add_parser("setup")
    setup.add_argument("--output", required=True, type=Path)
    metrics = subs.add_parser("metrics")
    metrics.add_argument("--state", type=Path, action="append", required=True)
    metrics.add_argument("--publish", action="store_true")
    metrics.add_argument("--output", required=True, type=Path)
    queue = subs.add_parser("queue")
    queue.add_argument("--state", type=Path, required=True)
    queue.add_argument("--evidence", type=Path, required=True)
    queue.add_argument("--queue-id", required=True)
    queue.add_argument("--family", required=True,
                       help="Same incident/reproduction family uses one split")
    queue.add_argument("--split", choices=["development", "holdout"], required=True)
    queue.add_argument("--output", required=True, type=Path)
    export = subs.add_parser("export-annotations")
    export.add_argument("--manifest", type=Path, action="append", required=True)
    export.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    settings = Settings()
    offline = args.command == "metrics" and not args.publish
    client = None if offline else client_for(settings, required=True)
    try:
        if args.command == "setup":
            result = bootstrap(client)
        elif args.command == "metrics":
            result = []
            for path in args.state:
                state = RunState.model_validate_json(path.read_text())
                result.append(run_metrics(state))
                if args.publish:
                    publish_run_metrics(client, state)
        elif args.command == "queue":
            state = RunState.model_validate_json(args.state.read_text())
            evidence = [Evidence.model_validate(e) for e in json.loads(
                args.evidence.read_text())]
            result = enqueue_reviews(
                client, args.queue_id, state, evidence,
                family=args.family, split=args.split,
            )
        else:
            entries = [e for path in args.manifest
                       for e in json.loads(path.read_text())["entries"]]
            completed, scores = set(), []
            for queue_id in sorted({e["queue_id"] for e in entries}):
                done, labels = collect_annotations(client, queue_id)
                completed.update(done)
                scores.extend(labels)
            result = annotation_dataset(entries, completed, scores)
            for split in ("development", "holdout"):
                items = result["datasets"].get(split, [])
                dump(args.output_directory / (split + ".json"), items)
            dump(args.output_directory / "coverage.json", {
                k: v for k, v in result.items() if k != "datasets"
            })
            print(json.dumps({k: v for k, v in result.items()
                              if k != "datasets"}, ensure_ascii=False))
            return
        dump(args.output, result)
        print("Saved", args.output)
    finally:
        if client:
            client.flush()


if __name__ == "__main__":
    main()
