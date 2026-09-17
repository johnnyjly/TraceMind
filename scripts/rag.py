"""Explicit preparation, bounded ingestion and retrieval of knowledge
snapshots.
"""

import argparse
import asyncio
import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path

import httpx

from tracemind.agent.runtime.rag_model_factory import create_rag_completion
from tracemind.config import Settings
from tracemind.rag.chunking import prepare_corpus
from tracemind.rag.config import RagSettings
from tracemind.rag.corpus import PARSER_VERSION, load_corpus
from tracemind.rag.engine import RagRuntime
from tracemind.rag.ingestion import (
    build_snapshot,
    snapshot_path,
    verify_snapshot,
)
from tracemind.rag.model_bridge import Completion
from tracemind.rag.local_models import LocalModels, load_tokenizer


def create_completion(
    settings: RagSettings, *, env_file: Path | list[Path] | None = None,
    callbacks=None, purpose="ingestion",
) -> Completion:
    """Assemble a lazy provider adapter outside the retrieval library."""
    provider_settings = Settings(
        **({"_env_file": env_file} if env_file else {}))
    return create_rag_completion(
        provider_settings, settings, callbacks=callbacks, purpose=purpose)


async def run(args) -> None:
    """Flush short-lived CLI traces on success and failure."""
    from tracemind.observability.langfuse import client_for
    client = client_for(Settings(_env_file=args.env_file))
    try:
        await _run(args)
    finally:
        if client is not None:
            await asyncio.to_thread(client.flush)


async def _run(args) -> None:
    """Execute one explicit setup, readiness, or retrieval operation."""
    settings = RagSettings(_env_file=args.env_file)
    from tracemind.observability.langfuse import client_for, callbacks_for
    app_settings = Settings(_env_file=args.env_file)
    client = client_for(app_settings)
    updates = {
        key: value
        for key, value in {
            "snapshot": args.snapshot,
            "directory": args.directory,
            "model_cache_directory": args.model_cache,
            "llm_max_calls": args.max_model_calls,
            "mode": args.mode,
        }.items()
        if value is not None
    }
    settings = RagSettings(
        _env_file=None, **{**settings.model_dump(), **updates}
    )
    if args.action in {"prepare", "build"}:
        corpus = await asyncio.to_thread(
            load_corpus,
            args.corpus,
            profile=args.profile,
            repositories={
                "tracemind": Path(__file__).resolve().parents[1],
                "opentelemetry-demo": args.demo_repo,
            },
            pdf_pipeline=settings.pdf_pipeline,
            artifacts_path=settings.docling_directory,
        )
        tokenizer = await asyncio.to_thread(load_tokenizer, settings)
        prepared = await asyncio.to_thread(
            prepare_corpus, corpus, tokenizer, settings
        )
        counts = {
            "documents": len(prepared),
            "parents": sum(len(p.parents) for p in prepared),
            "chunks": sum(len(p.chunks) for p in prepared),
        }
        if args.action == "prepare":
            output = (
                args.prepare_output
                or settings.directory / "prepared" / f"{args.profile}.json"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {
                        **counts,
                        "parser": PARSER_VERSION,
                        "profile": corpus.profile,
                        "manifest_sha256": corpus.manifest_sha256,
                        "sources_sha256": corpus.sources_sha256,
                        "source_checks": corpus.source_checks,
                        "documents_prepared": [
                            {
                                "metadata": p.document.metadata,
                                "normalized_text": p.document.text,
                                "parents": [
                                    parent.model_dump() for parent in p.parents
                                ],
                                "chunks": [c.model_dump() for c in p.chunks],
                            }
                            for p in prepared
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(
                json.dumps({**counts, "output": str(output), "model_calls": 0})
            )
            return
        runtime = RagRuntime(
            settings,
            completion=create_completion(
                settings,
                env_file=args.env_file,
                callbacks=callbacks_for(app_settings)),
            langfuse_client=client,
        )
        manifest = await build_snapshot(
            runtime, corpus, prepared, timeout_seconds=args.build_timeout
        )
        print(
            json.dumps(
                {
                    **counts,
                    "snapshot": settings.snapshot,
                    "state": manifest["state"],
                    "graph": manifest["graph"],
                    "model_usage": manifest["model_usage"],
                }
            )
        )
        return
    if args.action == "verify":
        manifest = verify_snapshot(settings.workspace)
        print(
            json.dumps(
                {
                    "snapshot": manifest["snapshot"],
                    "state": manifest["state"],
                    "chunks": manifest["chunks"],
                    "local_hashes_verified": True,
                    "qdrant_checked": False,
                }
            )
        )
        return
    if args.action == "models":
        models = LocalModels(settings)
        await asyncio.to_thread(models.load, download=True)
        if settings.pdf_pipeline == "standard":
            from docling.utils.model_downloader import download_models
            await asyncio.to_thread(
                download_models, output_dir=settings.docling_directory,
                with_code_formula=False, with_picture_classifier=False,
                with_rapidocr=False,
            )
        vectors = await models.embed(["A service request failed."])
        scores = await models.rank(
            "service failure",
            ["A service request failed.", "The garden has flowers."],
        )
        print(
            json.dumps(
                {
                    "embedding_shape": list(vectors.shape),
                    "reranker_scores": scores,
                    "model_revisions_pinned": True,
                }
            )
        )
        return
    if args.action == "check":
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(
                str(settings.qdrant_url).rstrip("/") + "/readyz"
            )
            response.raise_for_status()
        published = snapshot_path(settings.workspace).exists()
        if published:
            verify_snapshot(settings.workspace)
        elif (settings.workspace / "build.json").exists():
            raise ValueError("knowledge build is incomplete")
        database = settings.workspace / "keyword.sqlite3"
        with closing(
            sqlite3.connect(
                database.resolve().as_uri() + "?mode=ro",
                uri=True,
            )
        ) as db:
            count = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        print(
            json.dumps(
                {
                    "qdrant_ready": True,
                    "snapshot": settings.snapshot,
                    "chunks": count,
                    "corpus_published": published,
                }
            )
        )
        return
    if args.action == "evaluate":
        from tracemind.evaluation.budget import ModelCallBudget
        from tracemind.evaluation.experiments import (
            accepted, run_experiment, verify_publication,
        )
        from tracemind.evaluation.retrieval import RetrievalSuite
        suite = RetrievalSuite.model_validate_json(args.suite.read_text())
        if suite.snapshot != settings.snapshot:
            raise ValueError("evaluation snapshot mismatch")
        client = client_for(app_settings, required=True)
        if not client.auth_check():
            raise ValueError("Langfuse authentication failed")
        data = [{"input": {"query": q.query, "mode": args.mode},
                 "expected_output": {"relevance": q.relevant}}
                for q in suite.questions]
        # The SDK runner owns its event loop and each item's runtime.
        result = await asyncio.to_thread(
            run_experiment, app_settings, client, data,
            kind="retrieval", name="rag-" + suite.snapshot,
            budget=ModelCallBudget(args.max_model_calls),
            rag_settings=settings.model_copy(update={"result_limit": args.k}),
        )
        published = await asyncio.to_thread(verify_publication, client, result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "experiment_id": result.experiment_id,
            "published": published, "accepted": accepted(result),
        }, indent=2) + "\n")
        print(result.format())
        if not accepted(result) or not published:
            raise SystemExit(1)
        return
    runtime = RagRuntime(
        settings,
        completion=create_completion(settings, env_file=args.env_file,
                                     purpose="retrieval",
                                     callbacks=callbacks_for(app_settings)),
        langfuse_client=client,
    )
    try:
        await runtime.initialize()
        if args.action == "init":
            print(
                json.dumps(
                    {
                        "snapshot": settings.snapshot,
                        "workspace": str(settings.workspace),
                        "chunks": runtime.index.count,
                        "model_calls": runtime.model.calls,
                    }
                )
            )
        else:
            hits = await runtime.search(args.query, mode=args.mode)
            print(json.dumps([hit.model_dump() for hit in hits], indent=2))
    finally:
        await runtime.close()


def main() -> None:
    """Require an explicit build ID and whole-build paid-call budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "models",
            "init",
            "check",
            "prepare",
            "build",
            "verify",
            "query",
            "evaluate",
        ),
    )
    parser.add_argument("--snapshot")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--env-file", type=Path, action="append")
    parser.add_argument("--corpus", type=Path, default=Path("knowledge"))
    parser.add_argument("--profile", default="core")
    parser.add_argument(
        "--demo-repo", type=Path, default=Path("../opentelemetry-demo")
    )
    parser.add_argument("--prepare-output", type=Path)
    parser.add_argument("--max-model-calls", type=int)
    parser.add_argument("--build-timeout", type=float, default=1800)
    parser.add_argument("--query")
    parser.add_argument("--mode", choices=("mix", "naive"))
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    args.env_file = args.env_file or Path(".env")
    if args.action == "query" and not args.query:
        parser.error("query requires --query")
    if args.action == "evaluate" and (
        not args.suite or not args.output or not args.max_model_calls
        or args.max_model_calls < 1
    ):
        parser.error(
            "evaluate requires --suite, --output and --max-model-calls")
    if args.action == "build" and (
        not args.snapshot
        or args.snapshot == "bootstrap"
        or not args.max_model_calls
    ):
        parser.error("build requires a new --snapshot and --max-model-calls")
    if args.build_timeout <= 0:
        parser.error("--build-timeout must be positive")
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("tracemind.rag").setLevel(logging.INFO)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
