"""Shared review contracts and durable runtime verification policy."""

from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from tracemind.models.diagnosis import Diagnosis
from tracemind.models.evidence import Evidence
from tracemind.models.incident import Incident


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(
        min_length=1,
        description=(
            "A verbatim contiguous excerpt from diagnosis.root_cause or "
            "one diagnosis.remediation string, retaining qualifications "
            "and negation. Never invent or paraphrase an assertion."
        ),
    )
    kind: Literal["diagnostic", "uncertainty", "process"] = Field(
        default="diagnostic",
        description=(
            "diagnostic: incident facts/causal or efficacy claims, "
            "supported by telemetry IDs. "
            "process: an assertion of what a query/tool already returned, "
            "never a proposed check or imperative, supported by execution "
            "event sequences. "
            "uncertainty: explicit original abstention or evidential limits."
        ),
    )
    verdict: Literal["supported", "insufficient", "contradicted"]
    supporting_ids: list[UUID] = Field(
        default_factory=list,
        description=(
            "Evidence UUIDs copied from subject.evidence. A supported "
            "diagnostic claim MUST "
            "have at least one. Trace IDs and execution event sequences "
            "are not evidence UUIDs."
        ),
    )
    conflicting_ids: list[UUID] = Field(default_factory=list)
    supporting_event_sequences: list[int] = Field(
        default_factory=list,
        description=(
            "Sequences copied from execution_events. A supported process "
            "claim MUST "
            "have at least one relevant event sequence."
        ),
    )
    conflicting_event_sequences: list[int] = Field(default_factory=list)
    missing_observations: list[str] = Field(
        default_factory=list,
        description=(
            "Observations necessary to establish THIS exact claim. MUST "
            "be empty for supported "
            "diagnostic/process claims. If necessary observations are "
            "missing, verdict MUST be "
            "insufficient. Only supported uncertainty claims may list "
            "gaps justifying abstention."
        ),
    )
    rationale: str = Field(min_length=1)


class DiagnosisReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[Claim] = Field(
        min_length=1, max_length=16,
        description=(
            "Original assertions only. Exclude proposed checks, requests "
            "and imperatives entirely, including under kind=process. "
            "Include promised remediation effects as diagnostic assertions."
        ),
    )
    uncertainty_appropriate: bool
    root_cause_correctness: bool | None
    process_quality: float | None = Field(default=None, ge=0, le=1)
    rationale: str = Field(min_length=1)

    @property
    def verdict(self) -> str:
        verdicts = {claim.verdict for claim in self.claims}
        if "contradicted" in verdicts:
            return "contradicted"
        return "insufficient" if "insufficient" in verdicts else "supported"

    @property
    def accepted(self) -> bool:
        return self.verdict == "supported" and self.uncertainty_appropriate


class ReviewSubject(BaseModel):
    """Only observed diagnostic data; no held-out labels or model history."""

    incident: Incident | None = None
    diagnosis: Diagnosis
    evidence: list[Evidence]
    start: AwareDatetime
    end: AwareDatetime
    knowledge: list[dict] = Field(default_factory=list)
    execution_events: list[dict] = Field(default_factory=list)


class VerificationPolicy(BaseModel):
    provider: Literal["google", "openai", "deepseek"] = "deepseek"
    model: str = Field(default="deepseek-v4-pro", min_length=1)
    timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    input_bytes: int = Field(default=65536, ge=1)
    output_tokens: int = Field(default=4096, ge=1)


class VerificationRecord(BaseModel):
    """A saved candidate plus a single review attempt's durable outcome."""

    candidate: Diagnosis
    input_sha256: str
    rubric_version: str
    status: Literal["running", "completed", "failed"] = "running"
    review: DiagnosisReview | None = None
    error_code: str | None = None
