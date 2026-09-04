"""Fuse ranked retrieval branches using LangChain reciprocal ranks."""

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from tracemind.rag.models import KnowledgeChunk, RetrievalHit


def fuse_candidates(
    branches: dict[str, list[KnowledgeChunk]],
    *,
    limit: int,
) -> list[RetrievalHit]:
    """Fuse reciprocal ranks instead of adding incompatible backend scores."""
    hits: dict[str, RetrievalHit] = {}
    documents = []
    for branch, chunks in branches.items():
        seen = set()
        for chunk in chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            hit = hits.get(chunk.chunk_id)
            if hit is None:
                hit = RetrievalHit(chunk=chunk, branches=[], fusion_score=0)
                hits[chunk.chunk_id] = hit
            elif hit.chunk != chunk:
                raise ValueError("conflicting metadata for one chunk ID")
            hit.branches.append(branch)
        documents.append([
            Document(page_content=c.text, metadata={"chunk_id": c.chunk_id})
            for c in dict((c.chunk_id, c) for c in chunks).values()
        ])
    if not documents:
        return []
    ensemble = EnsembleRetriever(
        retrievers=[RunnableLambda(lambda query, rows=rows: rows)
                    for rows in documents],
        weights=[1.0] * len(documents), id_key="chunk_id",
    )
    ranked = ensemble.weighted_reciprocal_rank(documents)
    for position, document in enumerate(ranked):
        # Framework owns RRF. This ordinal is used only for deterministic ties.
        hits[document.metadata["chunk_id"]].fusion_score = 1 / (position + 1)
    return [hits[d.metadata["chunk_id"]] for d in ranked[:limit]]
