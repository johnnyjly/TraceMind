"""Local cutoff selection on development labels,
then frozen holdout scoring.
"""

import argparse
import asyncio
import json
import sqlite3
from hashlib import sha256
from pathlib import Path
from time import monotonic

from tracemind.evaluation.retrieval import RetrievalSuite
from tracemind.rag.config import RagSettings
from tracemind.rag.ingestion import verify_snapshot
from tracemind.rag.local_models import LocalModels


def evaluate(rows, cutoff):
    positives = [r for r in rows if any(v > 0 for v in r["relevant"].values())]
    negatives = [
        r for r in rows if not any(v > 0 for v in r["relevant"].values())
    ]
    coverage = irrelevant = relevant_found = relevant_total = empty = 0
    for row in rows:
        documents = {
            s["doc_id"] for s in row["scores"] if s["score"] >= cutoff
        }
        truth = {k for k, v in row["relevant"].items() if v > 0}
        coverage += bool(documents & truth)
        relevant_found += len(documents & truth)
        relevant_total += len(truth)
        irrelevant += len(documents - truth)
        empty += not documents and not truth
    return {
        "positive_query_coverage": coverage / len(positives)
        if positives
        else None,
        "relevant_document_recall": relevant_found / relevant_total
        if relevant_total
        else None,
        "negative_abstention": empty / len(negatives) if negatives else None,
        "irrelevant_document_admissions": irrelevant,
    }


async def run(args):
    dev = RetrievalSuite.model_validate_json(args.development.read_text())
    held = RetrievalSuite.model_validate_json(args.holdout.read_text())
    if held.snapshot != dev.snapshot:
        raise ValueError("calibration suites must use one snapshot")
    settings = RagSettings(
        _env_file=None,
        snapshot=dev.snapshot,
        directory=args.directory,
        model_cache_directory=args.model_cache,
    )
    manifest = verify_snapshot(settings.workspace)
    with sqlite3.connect(
        (settings.workspace / "keyword.sqlite3").resolve().as_uri()
        + "?mode=ro",
        uri=True,
    ) as db:
        chunks = [
            json.loads(r[0])
            for r in db.execute("select payload from chunks order by chunk_id")
        ]
    models = LocalModels(settings)
    groups = {}
    for split, suite in [("development", dev), ("holdout", held)]:
        rows = []
        for q in suite.questions:
            started = monotonic()
            scores = await models.rank(q.query, [c["text"] for c in chunks])
            rows.append(
                {
                    "id": q.id,
                    "query": q.query,
                    "relevant": q.relevant,
                    "elapsed_ms": (monotonic() - started) * 1000,
                    "scores": [
                        {
                            "doc_id": c["doc_id"],
                            "chunk_id": c["chunk_id"],
                            "score": s,
                        }
                        for c, s in zip(chunks, scores)
                    ],
                }
            )
        groups[split] = rows
        if split == "development":
            candidates = sorted(
                {0.0, *[s["score"] for r in rows for s in r["scores"]]}
            )
            sweep = [{"cutoff": t, **evaluate(rows, t)} for t in candidates]
            # Quality-first: retain a relevant document for every development
            # positive, then maximize abstention and minimize irrelevant hits.
            chosen = max(
                sweep,
                key=lambda r: (
                    r["positive_query_coverage"],
                    r["negative_abstention"],
                    -r["irrelevant_document_admissions"],
                    r["relevant_document_recall"],
                    -r["cutoff"],
                ),
            )
    report = {
        "snapshot": dev.snapshot,
        "snapshot_fingerprint": sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "reranker": settings.reranker_model,
        "revision": settings.reranker_revision,
        "cutoff": chosen["cutoff"],
        "selection": "development only; quality-first query coverage",
        "model_calls": 0,
        "candidate_scope": "all canonical chunks; retrieval recall evaluated "
        "separately",
        "suite_hashes": {
            str(p): sha256(p.read_bytes()).hexdigest()
            for p in [args.development, args.holdout]
        },
        "development": evaluate(groups["development"], chosen["cutoff"]),
        "holdout": evaluate(groups["holdout"], chosen["cutoff"]),
        "sweep": sweep,
        "queries": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"sweep", "queries"}}
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in [
        "development",
        "holdout",
        "directory",
        "model-cache",
        "output",
    ]:
        p.add_argument("--" + name, type=Path, required=True)
    asyncio.run(run(p.parse_args()))
