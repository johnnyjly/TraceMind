"""Bounded, durable knowledge references distinct from incident evidence."""

from hashlib import sha256

from pydantic import BaseModel, Field, computed_field

from tracemind.rag.models import RetrievalHit


class KnowledgePolicy(BaseModel):
    """Pin online retrieval to one published snapshot and retention budget."""

    snapshot: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    max_references: int = Field(default=20, ge=1, le=100)
    retention_bytes: int = Field(default=65536, ge=1)
    context_bytes: int = Field(default=8192, ge=64)
    result_limit: int = Field(default=5, ge=1, le=20)
    max_queries: int = Field(default=3, ge=1, le=3)
    model_output_tokens: int = Field(default=512, ge=1)
    reserved_model_calls: int = Field(default=1, ge=1)


class KnowledgeReference(BaseModel):
    """Preserve an actual retrieval result and immutable source identity."""

    snapshot: str
    hit: RetrievalHit

    @computed_field
    @property
    def id(self) -> str:
        """Bind citations to the snapshot, document revision and chunk text."""
        chunk = self.hit.chunk
        identity = "\0".join((
            self.snapshot, chunk.doc_id, chunk.chunk_id,
            chunk.document_sha256, chunk.source_revision,
            chunk.content_sha256,
        ))
        return "knowledge-" + sha256(identity.encode()).hexdigest()


def knowledge_preview(reference: KnowledgeReference) -> dict:
    """Return a bounded text preview with source provenance and a citation."""
    chunk = reference.hit.chunk
    return {
        "knowledge_ref": reference.id, "snapshot": reference.snapshot,
        "chunk_id": chunk.chunk_id, "title": chunk.title,
        "section": chunk.section, "text": chunk.text[:1200],
        "truncated": len(chunk.text) > 1200,
        "source_refs": chunk.source_refs,
        "source_revision": chunk.source_revision,
        "document_sha256": chunk.document_sha256,
    }
