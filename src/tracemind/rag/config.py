"""Explicit, environment-backed settings for a fixed RAG workspace."""

from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RagSettings(BaseSettings):
    """Configure optional RAG without activating it in diagnosis runs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TRACEMIND_RAG_",
        extra="ignore",
    )

    directory: Path = Path("artifacts/rag")
    model_cache_directory: Path | None = None
    snapshot: str = Field(default="bootstrap", pattern=r"^[a-zA-Z0-9_-]+$")
    qdrant_url: AnyHttpUrl = "http://127.0.0.1:6333"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_revision: str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    embedding_dimension: int = Field(default=384, ge=1)
    embedding_max_tokens: int = Field(default=512, ge=1)
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_revision: str = "2cfc18c9415c912f9d8155881c133215df768a70"
    reranker_max_tokens: int = Field(default=512, ge=1)
    embedding_query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )
    device: Literal["cpu", "mps"] = "cpu"
    batch_size: int = Field(default=8, ge=1, le=64)
    chunk_tokens: int = Field(default=320, ge=1)
    pdf_pipeline: Literal["standard", "native"] = "standard"
    mode: Literal["mix", "naive"] = "mix"
    candidate_limit: int = Field(default=20, ge=1, le=100)
    result_limit: int = Field(default=5, ge=1, le=20)
    bm25_enabled: bool = True
    llm_provider: str | None = Field(default=None, min_length=1)
    llm_model: str | None = Field(default=None, min_length=1)
    llm_max_calls: int = Field(default=4, ge=1)
    llm_output_tokens: int = Field(default=2048, ge=1)
    timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    retrieval_context_tokens: int = Field(default=8192, ge=1)
    summary_max_tokens: int = Field(default=192, ge=1)
    summary_length_recommended: int = Field(default=128, ge=1)

    @model_validator(mode="after")
    def validate_sizes(self) -> "RagSettings":
        """Keep chunking and retrieval bounds internally consistent."""
        if self.chunk_tokens + 32 > self.embedding_max_tokens:
            raise ValueError("chunk size must reserve embedding prefix space")
        if self.result_limit > self.candidate_limit:
            raise ValueError("result limit exceeds candidate limit")
        if self.summary_length_recommended > self.summary_max_tokens:
            raise ValueError("recommended summary length exceeds summary cap")
        return self

    @property
    def workspace(self) -> Path:
        """Return the directory for one fixed snapshot identifier."""
        return self.directory / "snapshots" / self.snapshot

    @property
    def models_directory(self) -> Path:
        """Allow independent snapshots to reuse cached, pinned model
        weights.
        """
        return self.model_cache_directory or self.directory / "models"

    @property
    def docling_directory(self) -> Path:
        """Keep PDF layout models outside published knowledge snapshots."""
        return self.models_directory / "docling"
