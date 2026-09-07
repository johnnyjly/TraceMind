"""Section-aware preparation, bounded ingestion and snapshot publication."""

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from tracemind.rag.corpus import PARSER_VERSION, Corpus, digest

if TYPE_CHECKING:
    from tracemind.rag.chunking import PreparedDocument

logger = logging.getLogger(__name__)


def snapshot_path(workspace: Path) -> Path:
    return workspace / "snapshot.json"


def verify_snapshot(workspace: Path) -> dict:
    """Reject incomplete or altered local artifacts before opening a reader."""
    manifest = json.loads(snapshot_path(workspace).read_text())
    if manifest.get("state") != "published" or not manifest.get("files"):
        raise ValueError("knowledge snapshot is not published")
    for relative, expected in manifest["files"].items():
        path = (workspace / relative).resolve()
        if not path.is_relative_to(workspace.resolve()):
            raise ValueError("snapshot artifact escapes workspace")
        if digest(path.read_bytes()) != expected:
            raise ValueError(f"snapshot artifact hash mismatch: {relative}")
    return manifest


async def build_snapshot(runtime, corpus, prepared, *, timeout_seconds):
    from tracemind.observability.langfuse import observation

    with observation(
        getattr(runtime, "langfuse_client", None),
        "knowledge.ingestion",
        input={
            "snapshot": runtime.settings.snapshot,
            "documents": len(prepared),
        },
    ) as span:
        manifest = await _build_snapshot(
            runtime, corpus, prepared, timeout_seconds=timeout_seconds
        )
        if span is not None:
            span.update(output=manifest)
        return manifest


async def _build_snapshot(
    runtime,
    corpus: Corpus,
    prepared: tuple["PreparedDocument", ...],
    *,
    timeout_seconds: float,
) -> dict:
    """Publish only after all documents, canonical text and indexes agree.

    Failed work remains unpublished for inspection. A retry uses a new snapshot
    ID; this first build workflow deliberately does not repair partial graphs.
    The injected runtime model owns one call budget for the entire build.
    """
    workspace = runtime.settings.workspace
    if workspace.exists():
        raise ValueError("build requires a new snapshot ID")
    if not prepared or tuple(p.document for p in prepared) != corpus.documents:
        raise ValueError("build requires the complete selected corpus")
    lookup = {digest(p.document.text.encode()): p for p in prepared}
    if len(lookup) != len(prepared):
        raise ValueError(
            "duplicate extracted documents require explicit deduplication"
        )
    workspace.mkdir(parents=True, exist_ok=False)
    record = {
        "state": "building",
        "profile": corpus.profile,
        "manifest_sha256": corpus.manifest_sha256,
        "sources_sha256": corpus.sources_sha256,
        "parser": PARSER_VERSION,
        "documents": [
            {
                "doc_id": p.document.metadata["doc_id"],
                "sha256": p.document.metadata["sha256"],
                "extracted_sha256": p.document.extracted_sha256,
                "parents": len(p.parents),
                "chunks": len(p.chunks),
            }
            for p in prepared
        ],
    }
    build_path = workspace / "build.json"

    def save_progress():
        record["model_usage"] = {
            "calls": runtime.model.calls,
            "input_tokens": runtime.model.input_tokens,
            "output_tokens": runtime.model.output_tokens,
            "unreported_calls": runtime.model.unreported_calls,
        }
        build_path.write_text(json.dumps(record, indent=2) + "\n")

    save_progress()

    def chunker(tokenizer, content, *args, **kwargs):
        doc = lookup[digest(content.encode())]
        return [
            {
                "chunk_id": c.chunk_id,
                "content": c.text,
                "tokens": len(tokenizer.encode(c.text)),
                "chunk_order_index": n,
                "content_headings": c.section,
            }
            for n, c in enumerate(doc.chunks)
        ]

    try:
        async with asyncio.timeout(timeout_seconds):
            await runtime.initialize(chunking_func=chunker)
            await runtime.require_empty_vectors()
            runtime.model.reset()
            all_chunks = [c for p in prepared for c in p.chunks]
            all_parents = [parent for p in prepared for parent in p.parents]
            runtime.index.register(all_chunks, parents=all_parents)
            for n, p in enumerate(prepared, 1):
                doc = p.document
                logger.info(
                    "rag.build snapshot=%s document=%s "
                    "progress=%s/%s chunks=%s",
                    runtime.settings.snapshot,
                    doc.metadata["doc_id"],
                    n,
                    len(prepared),
                    len(p.chunks),
                )
                await runtime.rag.apipeline_enqueue_documents(
                    [doc.text],
                    ids=[doc.key],
                    file_paths=[doc.metadata["path"]],
                    process_options="C",
                )
                await runtime.rag.apipeline_process_enqueue_documents()
                status = await runtime.rag.doc_status.get_by_id(doc.key)
                expected = {c.chunk_id for c in p.chunks}
                if (
                    not status
                    or status.get("status") != "processed"
                    or set(status.get("chunks_list", [])) != expected
                ):
                    raise RuntimeError(
                        f"LightRAG ingestion incomplete: {
                            doc.metadata['doc_id']
                        }"
                    )
                await runtime.rag.chunks_vdb.index_done_callback()
                vectors = await runtime.rag.chunks_vdb.get_by_ids(
                    [c.chunk_id for c in p.chunks]
                )
                if len(vectors) != len(p.chunks) or any(
                    not row or row.get("content") != c.text
                    for row, c in zip(vectors, p.chunks)
                ):
                    raise ValueError(
                        "Qdrant chunks differ from canonical chunks"
                    )
                for chunk in p.chunks:
                    stored = await runtime.rag.text_chunks.get_by_id(
                        chunk.chunk_id
                    )
                    if not stored or stored.get("content") != chunk.text:
                        raise ValueError(
                            "LightRAG text differs from canonical chunk"
                        )
                    if runtime.index.get(chunk.chunk_id) != chunk:
                        raise ValueError(
                            "keyword metadata differs from canonical chunk"
                        )
                record["completed_documents"] = n
                save_progress()
            if runtime.index.count != len(all_chunks):
                raise ValueError("canonical chunk count mismatch")
            graph = runtime.rag.chunk_entity_relation_graph
            nodes, edges = (
                await graph.get_all_nodes(),
                await graph.get_all_edges(),
            )
            known_ids = {c.chunk_id for c in all_chunks}
            for item in [*nodes, *edges]:
                if not item.get("source_id") or not set(
                    item["source_id"].split("<SEP>")
                ).issubset(known_ids):
                    raise ValueError("graph has unregistered source chunks")
            record["graph"] = {"nodes": len(nodes), "edges": len(edges)}
            await runtime.close()
            (workspace / "corpus_manifest.json").write_bytes(
                corpus.manifest_bytes
            )
            (workspace / "source_registry.json").write_bytes(
                corpus.source_bytes
            )
            (workspace / "sources.json").write_text(
                json.dumps(
                    {
                        "sources": corpus.sources,
                        "checks": corpus.source_checks,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )
            record["state"] = "complete"
            save_progress()
            files = {
                str(path.relative_to(workspace)): digest(path.read_bytes())
                for path in workspace.rglob("*")
                if path.is_file()
            }
            published = {
                **record,
                "state": "published",
                "chunks": len(all_chunks),
                "parents": len(all_parents),
                "snapshot": runtime.settings.snapshot,
                "files": files,
            }
            pending = workspace / "snapshot.json.tmp"
            pending.write_text(json.dumps(published, indent=2) + "\n")
            pending.replace(snapshot_path(workspace))
            return published
    except BaseException as exc:
        record["state"] = "failed"
        record["error_type"] = type(exc).__name__
        save_progress()
        raise
    finally:
        await runtime.close()
