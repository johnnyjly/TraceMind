"""Run the local SQLite-backed diagnosis worker."""

import argparse
import asyncio
import logging

from tracemind.agent.worker import RunWorker
from tracemind.config import get_settings


def main() -> None:
    """Poll persistent runs until interrupted, or process one claim."""
    parser = argparse.ArgumentParser(description="Run the TraceMind worker")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    worker = RunWorker(get_settings())
    if args.once:
        asyncio.run(worker.run_once())
    else:
        asyncio.run(worker.run_forever())


if __name__ == "__main__":
    main()
