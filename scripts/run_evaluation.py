import argparse
import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tracemind.config import get_settings
from tracemind.evaluation import (
    EvaluationCase, EvaluationReport, run_live_evaluation,
)
from tracemind.models.memory import MemorySnapshot


logger = logging.getLogger("tracemind.evaluation.cli")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_PATH = PROJECT_ROOT / "evals" / "cases" / "payment_failure.json"


async def run(
    case_path: Path,
    lookback_minutes: int,
    output_path: Path,
    memory_snapshot_path: Path | None = None,
) -> EvaluationReport:
    case = EvaluationCase.load(case_path)
    end = datetime.now(UTC)
    start = end - timedelta(minutes=lookback_minutes)
    report = await run_live_evaluation(
        case,
        get_settings(),
        start=start,
        end=end,
        memory_snapshot=(MemorySnapshot.model_validate_json(
            memory_snapshot_path.read_text())
            if memory_snapshot_path else None),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2) + "\n")
    logger.info("Evaluation passed: %s",
                report.result.passed if report.result else False)
    logger.info("Report: %s", output_path.resolve())
    return report


def configure_logging(*, verbose: bool) -> None:
    """Configure CLI output without enabling third-party debug payloads."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s "
        "[run=%(run_id)s parent=%(parent_run_id)s] %(message)s",
        defaults={"run_id": "-", "parent_run_id": "-"},
    ))
    logging.basicConfig(level=logging.WARNING, handlers=[handler])
    logging.getLogger("tracemind").setLevel(logging.INFO)
    logging.getLogger("tracemind.agent.harness").setLevel(
        logging.DEBUG if verbose else logging.INFO
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an TraceMind evaluation case"
    )
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE_PATH)
    parser.add_argument("--lookback-minutes", type=int, default=10)
    parser.add_argument("--memory-snapshot", type=Path,
                        help="Frozen JSON corpus captured before this window")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log sanitized run, model, and tool events to the terminal",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT / "artifacts" / "evaluations" / "payment-failure.json"
        ),
    )
    args = parser.parse_args()
    if args.lookback_minutes < 1:
        parser.error("--lookback-minutes must be at least 1")
    configure_logging(verbose=args.verbose)
    report = asyncio.run(
        run(
            args.case,
            args.lookback_minutes,
            args.output,
            args.memory_snapshot,
        )
    )
    raise SystemExit(0 if report.result and report.result.passed else 1)


if __name__ == "__main__":
    main()
