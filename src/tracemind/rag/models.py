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
    source_items: list[str] = Field(default_factory=list)
    parser_version: str = "legacy-v1"


class KnowledgeParent(KnowledgeSource):
    """Complete normalized section; never embedded as an oversized child."""

    parent_id: str = Field(min_length=1)
    source_start: int = Field(ge=0)
    source_end: int = Field(gt=0)


class KnowledgeChunk(KnowledgeSource):
    """A bounded child with source pointers, or legacy character offsets."""

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
    fusion_score: float = Field(description=(
        "Reciprocal position in the framework-fused list. "
        "Legacy records may contain raw RRF scores."
    ))
    rerank_score: float | None = None
    parent: KnowledgeParent | None = None
