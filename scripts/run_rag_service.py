"""Run a prewarmed local retrieval service separately from API and worker."""

import os
import logging

# Bound local CPU inference before importing the numerical libraries.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", action="append", type=Path)
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    import torch
    import uvicorn
    from tracemind.rag.query_service import create_app
    from tracemind.config import Settings, settings_files
    from tracemind.rag.config import RagSettings
    from tracemind.observability.langfuse import client_for

    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    uvicorn.run(
        create_app(
            RagSettings(_env_file=settings_files(args.env_file)),
            langfuse_client=client_for(
                Settings(_env_file=settings_files(args.env_file))
            ),
        ),
        host="127.0.0.1",
        port=args.port,
    )


if __name__ == "__main__":
    main()
