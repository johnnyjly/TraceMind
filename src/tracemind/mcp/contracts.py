"""Client-side validation of remote reply envelopes."""


from pydantic import BaseModel, ConfigDict, Field, model_validator

from tracemind.models import Evidence


class RemoteFailure(BaseModel):
    """Transfer safe retry classification, excluding backend error text."""

    model_config = ConfigDict(extra="forbid")
    category: str
    retryable: bool = False
    status_code: int | None = None
    retry_after: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class TelemetryReply(BaseModel):
    """Return exactly one observation list, metadata object or safe failure."""

    model_config = ConfigDict(extra="forbid")
    evidence: list[Evidence] | None = None
    metadata: dict | None = None
    error: RemoteFailure | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "TelemetryReply":
        """Prevent metadata or failure records from becoming observations."""
        if sum(item is not None for item in (
            self.evidence, self.metadata, self.error,
        )) != 1:
            raise ValueError("reply requires exactly one payload")
        return self
