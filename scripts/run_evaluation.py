import argparse
import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tracemind.config import get_settings
from tracemind.evaluation import EvaluationCase, run_live_evaluation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_PATH = PROJECT_ROOT / "evals" / "cases" / "payment_failure.json"


async def run(case_path: Path, lookback_minutes: int, output_path: Path) -> None:
    case = EvaluationCase.load(case_path)
    end = datetime.now(UTC)
    start = end - timedelta(minutes=lookback_minutes)
    report = await run_live_evaluation(
        case,
        get_settings(),
        start=start,
        end=end,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2) + "\n")
    print(f"Evaluation passed: {report.result.passed}")
    print(f"Report: {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an TraceMind evaluation case")
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE_PATH)
    parser.add_argument("--lookback-minutes", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "evaluations" / "payment-failure.json",
    )
    args = parser.parse_args()
    if args.lookback_minutes < 1:
        parser.error("--lookback-minutes must be at least 1")
    asyncio.run(run(args.case, args.lookback_minutes, args.output))


if __name__ == "__main__":
    main()
