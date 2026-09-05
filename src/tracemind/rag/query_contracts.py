"""Bounded internal contract for the read-only retrieval service."""
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tracemind.rag.models import RetrievalHit

Term = Annotated[str, Field(min_length=1, max_length=512)]


class QueryKeywords(BaseModel):
    model_config = ConfigDict(extra="forbid")
    high_level_keywords: list[Term] = Field(
        default_factory=list, max_length=32)
    low_level_keywords: list[Term] = Field(default_factory=list, max_length=32)


class PreparedQuery(QueryKeywords):
    query: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def require_keywords(self):
        if not self.high_level_keywords and not self.low_level_keywords:
            raise ValueError("prepared queries require keywords")
        return self


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot: str = Field(pattern=r"^[a-zA-Z0-9_-]+$", max_length=120)
    question: str = Field(min_length=1, max_length=12000)
    queries: list[PreparedQuery] = Field(min_length=1, max_length=4)


class RetrievalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot: str
    hits: list[RetrievalHit]
