"""Run committed telemetry replays and write a batch metrics report."""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from tracemind.config import get_settings
from tracemind.evaluation.replay import ReplayFixture, run_replay_evaluation
from tracemind.evaluation.models import EvaluationReport
from tracemind.models.memory import MemorySnapshot
from tracemind.evaluation.summary import (
    BatchReport, TokenPricing, build_batch_report,
)


logger = logging.getLogger("tracemind.evaluation.suite")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


async def run(paths: list[Path], output: Path,
              pricing: dict[str, TokenPricing], *,
              memory_snapshot: MemorySnapshot | None = None,
              compare_memory: bool = False) -> BatchReport:
    """Execute each fixture independently and retain every run outcome."""
    settings = get_settings()
    reports = []
    if compare_memory and memory_snapshot is None:
        raise ValueError("memory comparison requires a frozen snapshot")
    modes = ([None, memory_snapshot] if compare_memory else [memory_snapshot])
    for path in paths:
        fixture = ReplayFixture.load(path)
        for snapshot in modes:
            report = await run_replay_evaluation(
                fixture, settings, memory_snapshot=snapshot,
            )
            reports.append(report)
            logger.info("%s [%s]: %s", report.case_id, report.memory_mode,
                        report.error_type or
                        ("passed" if report.result.passed
                         else "failed scoring"))
    batch = build_batch_report(reports, pricing=pricing)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(batch.model_dump_json(indent=2) + "\n")
    return batch


def main() -> None:
    """Run an offline suite; explicit prices never imply known token usage."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fixtures", type=Path,
                        default=PROJECT_ROOT / "evals" / "replays")
    source.add_argument("--reports", type=Path, nargs="+",
                        help="Aggregate existing individual live reports")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "artifacts" / "evaluations"
                        / "replay-suite.json")
    parser.add_argument("--pricing", type=Path,
                        help="JSON provider:model map of USD token rates")
    parser.add_argument("--memory-snapshot", type=Path,
                        help="Explicit frozen JSON corpus; never the live DB")
    parser.add_argument("--compare-memory", action="store_true",
                        help="Pair memory-off/on replays with equal limits")
    args = parser.parse_args()
    if args.compare_memory and not args.memory_snapshot:
        parser.error("--compare-memory requires --memory-snapshot")
    if args.reports and (args.memory_snapshot or args.compare_memory):
        parser.error("memory options apply to fixtures, not existing reports")
    snapshot = (MemorySnapshot.model_validate_json(
        args.memory_snapshot.read_text()) if args.memory_snapshot else None)
    paths = sorted(args.fixtures.glob("*.json")) if not args.reports else []
    if not paths and not args.reports:
        parser.error("no replay fixtures found")
    pricing = ({key: TokenPricing.model_validate(value) for key, value in
                json.loads(args.pricing.read_text()).items()}
               if args.pricing else {})
    logging.basicConfig(level=logging.INFO)
    if args.reports:
        batch = build_batch_report([
            EvaluationReport.model_validate_json(path.read_text())
            for path in args.reports
        ], pricing=pricing)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(batch.model_dump_json(indent=2) + "\n")
        passed = batch.summary.passed
    else:
        batch = asyncio.run(run(paths, args.output, pricing,
                                memory_snapshot=snapshot,
                                compare_memory=args.compare_memory))
        passed = batch.summary.regression_passed
    logger.info("%d/%d gates passed; report: %s",
                passed, batch.summary.runs, args.output)
    raise SystemExit(0 if passed == batch.summary.runs else 1)


if __name__ == "__main__":
    main()
