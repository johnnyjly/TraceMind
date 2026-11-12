"""Immutable, content-checked prompt selected for one investigation."""

from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PromptSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: int = Field(ge=1)
    label: str
    content: str = Field(min_length=1)
    sha256: str
    contract_version: str
    source: Literal["langfuse"] = "langfuse"

    @model_validator(mode="after")
    def check_content(self):
        if sha256(self.content.encode()).hexdigest() != self.sha256:
            raise ValueError("prompt snapshot hash mismatch")
        return self
