"""Knowledge references and retrieval results distinct from observations."""

from hashlib import sha256

from pydantic import BaseModel, Field


class KnowledgeSource(BaseModel):
    """Pinned provenance shared by a section and its retrieval chunks."""

    doc_id: str = Field(min_length=1)
    title: str
    section: str
    text: str = Field(min_length=1)
    document_path: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_refs: list[str] = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    scope: str = ""
    heading_path: list[str] = Field(default_factory=list)
    extracted_sha256: str = ""


class KnowledgeParent(KnowledgeSource):
    """Complete normalized section; never embedded as an oversized child."""

    parent_id: str = Field(min_length=1)
    source_start: int = Field(ge=0)
    source_end: int = Field(gt=0)


class KnowledgeChunk(KnowledgeSource):
    """A bounded child with offsets into its complete parent section."""

    chunk_id: str = Field(min_length=1)
    parent_id: str | None = None
    source_start: int = Field(default=0, ge=0)
    source_end: int = Field(default=0, ge=0)
    source_text: str = ""
    identifiers: str = ""

    @property
    def content_sha256(self) -> str:
        """Hash the canonical text for retrieval consistency checks."""
        return sha256(self.text.encode()).hexdigest()


class RetrievalHit(BaseModel):
    """Ranked reference material; never an observation of the current fault."""

    chunk: KnowledgeChunk
    branches: list[str]
    fusion_score: float
    rerank_score: float | None = None


def fuse_candidates(
    branches: dict[str, list[KnowledgeChunk]],
    *,
    limit: int,
) -> list[RetrievalHit]:
    """Fuse reciprocal ranks instead of adding incompatible backend scores."""
    hits: dict[str, RetrievalHit] = {}
    for branch, chunks in branches.items():
        seen = set()
        for rank, chunk in enumerate(chunks, 1):
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
            hit.fusion_score += 1 / (60 + rank)
    return sorted(
        hits.values(), key=lambda h: (-h.fusion_score, h.chunk.chunk_id)
    )[:limit]
