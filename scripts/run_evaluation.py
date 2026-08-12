import argparse
import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tracemind.agent import ConsoleTraceCallback
from tracemind.config import get_settings
from tracemind.evaluation import EvaluationCase, run_live_evaluation


logger = logging.getLogger("tracemind.evaluation.cli")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_PATH = PROJECT_ROOT / "evals" / "cases" / "payment_failure.json"


async def run(
    case_path: Path,
    lookback_minutes: int,
    output_path: Path,
    *,
    verbose: bool,
    verbose_max_chars: int,
) -> None:
    case = EvaluationCase.load(case_path)
    end = datetime.now(UTC)
    start = end - timedelta(minutes=lookback_minutes)
    report = await run_live_evaluation(
        case,
        get_settings(),
        start=start,
        end=end,
        callbacks=(
            [ConsoleTraceCallback(max_chars=verbose_max_chars)]
            if verbose else None
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2) + "\n")
    logger.info("Evaluation passed: %s", report.result.passed)
    logger.info("Report: %s", output_path.resolve())


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
    logging.getLogger("tracemind.agent.console_trace").setLevel(
        logging.DEBUG if verbose else logging.INFO
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an TraceMind evaluation case"
    )
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE_PATH)
    parser.add_argument("--lookback-minutes", type=int, default=10)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print model messages and tool activity to the terminal",
    )
    parser.add_argument(
        "--verbose-max-chars",
        type=int,
        default=12_000,
        help="Maximum characters printed for each verbose event",
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
    if args.verbose_max_chars < 1:
        parser.error("--verbose-max-chars must be at least 1")
    configure_logging(verbose=args.verbose)
    asyncio.run(
        run(
            args.case,
            args.lookback_minutes,
            args.output,
            verbose=args.verbose,
            verbose_max_chars=args.verbose_max_chars,
        )
    )


if __name__ == "__main__":
    main()
