"""Completed diagnosis and its execution evidence."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from tracemind.models import Diagnosis, DiagnosisStatus, Evidence
from tracemind.models.memory import FeedbackUpdate
from tracemind.models.run import RunState


class DiagnosisRun(BaseModel):
    """Diagnosis result bundled with evidence, tool counts, and optional
    state.
    """

    diagnosis: Diagnosis
    evidence: list[Evidence]
    tool_calls: dict[str, int]
    state: RunState | None = None

    @property
    def tool_call_count(self) -> int:
        """Return the total admitted tool calls reported for this run."""
        return sum(self.tool_calls.values())


class DiagnosisDraft(BaseModel):
    """Resolved internal result with original public citation identifiers."""

    status: DiagnosisStatus
    root_cause: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[UUID] = Field(default_factory=list)
    knowledge_refs: list[str] = Field(default_factory=list)
    remediation: list[str] = Field(default_factory=list)
    feedback_updates: list[FeedbackUpdate] = Field(
        default_factory=list, max_length=3,
    )


class ProposalStatement(BaseModel):
    """One scoped statement with model-facing run-local references."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1400)
    evidence_refs: list[str] = Field(default_factory=list)
    knowledge_refs: list[str] = Field(default_factory=list)


class DiagnosisProposal(BaseModel):
    """Model output; application code numbers, resolves and renders it."""

    model_config = ConfigDict(extra="forbid")
    status: DiagnosisStatus
    confidence: float = Field(ge=0, le=1)
    findings: list[ProposalStatement] = Field(max_length=6)
    hypotheses: list[ProposalStatement] = Field(max_length=3)
    limitations: list[ProposalStatement] = Field(max_length=3)
    proposed_checks: list[ProposalStatement] = Field(max_length=3)
    feedback_updates: list[FeedbackUpdate] = Field(
        default_factory=list, max_length=3,
    )
