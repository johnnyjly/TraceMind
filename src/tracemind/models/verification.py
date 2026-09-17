"""Shared review contracts and durable runtime verification policy."""

from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from tracemind.models.diagnosis import Diagnosis
from tracemind.models.evidence import Evidence
from tracemind.models.incident import Incident


class ReviewItem(BaseModel):
    """Application-numbered text; model classifications are not trusted."""

    model_config = ConfigDict(extra="forbid")
    id: str
    section: Literal["finding", "hypothesis", "limitation", "proposed_check"]
    text: str = Field(min_length=1)
    evidence_ids: list[UUID] = Field(default_factory=list)
    knowledge_refs: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: str = ""
    text: str = Field(min_length=1)
    kind: Literal["diagnostic", "uncertainty", "process", "proposal"]
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


class ReferenceComponent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    matches: bool
    item_ids: list[str] = Field(max_length=12)
    rationale: str = Field(min_length=1, max_length=500)


class ReferenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    owner: ReferenceComponent
    operation: ReferenceComponent
    mechanism: ReferenceComponent
    propagation: ReferenceComponent
    exclusions: ReferenceComponent

    @property
    def correct(self) -> bool:
        return all(getattr(self, k).matches for k in type(self).model_fields)


class DiagnosisReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[Claim] = Field(
        min_length=1,
        max_length=32,
        description=(
            "Exactly one assessment per application-numbered item, including "
            "hypotheses, limitations and proposed checks."
        ),
    )
    uncertainty_appropriate: bool
    root_cause_correctness: bool | None = Field(
        description=(
            "Null unless an explicit held-out reference_answer or "
            "root_cause_service is supplied in reference. Observations and "
            "the original diagnosis are not held-out references."
        )
    )
    process_quality: float | None = Field(default=None, ge=0, le=1)
    reference_assessment: ReferenceAssessment | None = None
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
    items: list[ReviewItem] = Field(default_factory=list)
    start: AwareDatetime
    end: AwareDatetime
    knowledge: list[dict] = Field(default_factory=list)
    execution_events: list[dict] = Field(default_factory=list)


class VerificationPolicy(BaseModel):
    provider: Literal["google", "openai", "deepseek"] = "deepseek"
    model: str = Field(default="deepseek-v4-pro", min_length=1)
    deepseek_thinking: bool = True
    deepseek_reasoning_effort: Literal["low", "high", "max"] = "high"
    timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    input_bytes: int = Field(default=131072, ge=1)
    output_tokens: int = Field(default=4096, ge=1)


class VerificationRecord(BaseModel):
    """A saved candidate plus a single review attempt's durable outcome."""

    candidate: Diagnosis
    items: list[ReviewItem] = Field(default_factory=list)
    attempts: int = 0
    validation_errors: list[str] = Field(default_factory=list)
    input_sha256: str
    rubric_version: str
    status: Literal["running", "completed", "failed"] = "running"
    review: DiagnosisReview | None = None
    error_code: str | None = None
