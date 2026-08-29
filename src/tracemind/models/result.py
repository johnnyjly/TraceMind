"""Completed diagnosis and its execution evidence."""

from uuid import UUID

from pydantic import BaseModel, Field

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
    """Diagnostic proposal with status, confidence, citations, and
    remediation.
    """

    status: DiagnosisStatus
    root_cause: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[UUID] = Field(default_factory=list)
    remediation: list[str] = Field(default_factory=list)
    feedback_updates: list[FeedbackUpdate] = Field(
        default_factory=list, max_length=3,
    )
