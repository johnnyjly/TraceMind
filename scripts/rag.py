"""Set up and inspect RAG infrastructure without importing the corpus."""

import argparse
import asyncio
from contextlib import closing
import json
import logging
import sqlite3

import httpx

from tracemind.rag.config import RagSettings
from tracemind.rag.engine import RagRuntime
from tracemind.rag.local_models import LocalModels
from tracemind.rag.llm import Completion


def create_completion(settings: RagSettings) -> Completion:
    """Assemble a lazy provider adapter outside the retrieval library."""
    model = None

    async def complete(messages, *, max_tokens, keyword_extraction):
        """Create the selected client once, only when a model is needed."""
        nonlocal model
        if model is None:
            from tracemind.agent.llm import create_chat_model
            from tracemind.config import Settings

            provider_settings = Settings(
                llm_provider=settings.llm_provider,
                llm_model=settings.llm_model)
            model = create_chat_model(provider_settings)
        target = model
        if (keyword_extraction
                and settings.llm_provider in {"openai", "deepseek"}):
            target = model.bind(response_format={"type": "json_object"})
        return await target.ainvoke(messages, max_tokens=max_tokens)

    return complete


async def run(args) -> None:
    """Execute one explicit setup, readiness, or retrieval operation."""
    settings = RagSettings()
    if args.snapshot:
        settings = RagSettings(**{
            **settings.model_dump(), "snapshot": args.snapshot})
    if args.action == "models":
        models = LocalModels(settings)
        await asyncio.to_thread(models.load, download=True)
        vectors = await models.embed(["A service request failed."])
        scores = await models.rank("service failure", [
            "A service request failed.", "The garden has flowers."])
        print(json.dumps({"embedding_shape": list(vectors.shape),
                          "reranker_scores": scores,
                          "model_revisions_pinned": True}))
        return
    if args.action == "check":
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(
                str(settings.qdrant_url).rstrip("/") + "/readyz")
            response.raise_for_status()
        manifest = json.loads(
            (settings.workspace / "runtime.json").read_text())
        database = settings.workspace / "keyword.sqlite3"
        with closing(sqlite3.connect(
            database.resolve().as_uri() + "?mode=ro", uri=True,
        )) as db:
            count = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        print(json.dumps({"qdrant_ready": True,
                          "snapshot": settings.snapshot, "chunks": count,
                          "corpus_published": manifest["corpus_published"]}))
        return
    runtime = RagRuntime(settings, completion=create_completion(settings))
    try:
        await runtime.initialize()
        if args.action == "init":
            print(json.dumps({"snapshot": settings.snapshot,
                              "workspace": str(settings.workspace),
                              "chunks": runtime.index.count,
                              "model_calls": runtime.model.calls}))
        else:
            hits = await runtime.search(args.query, mode=args.mode)
            print(json.dumps([hit.model_dump() for hit in hits], indent=2))
    finally:
        await runtime.close()


def main() -> None:
    """Parse an explicit command; never start diagnosis or ingest documents."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("models", "init", "check", "query"))
    parser.add_argument("--snapshot")
    parser.add_argument("--query")
    parser.add_argument("--mode", choices=("mix", "naive"))
    args = parser.parse_args()
    if args.action == "query" and not args.query:
        parser.error("query requires --query")
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("tracemind.rag").setLevel(logging.INFO)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
