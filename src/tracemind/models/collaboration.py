"""Bounded task ledger and explicit evidence-bearing agent handoffs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tracemind.models.diagnosis import Diagnosis
from tracemind.models.budget import TokenLedger
from tracemind.models.verification import VerificationRecord, Claim


class InvestigationTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["telemetry", "knowledge"]
    question: str = Field(min_length=1, max_length=1000)
    hypothesis_ids: list[str] = Field(default_factory=list, max_length=4)
    distinguishing_observations: list[str] = Field(
        default_factory=list, max_length=3
    )
    acceptance_criteria: list[str] = Field(default_factory=list, max_length=4)
    context_refs: list[str] = Field(default_factory=list, max_length=16)
    requested_tools: list[str] = Field(default_factory=list, max_length=12)


class TaskInput(BaseModel):
    """Application-frozen handoff, never a parent's conversation history."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    criteria: dict[str, str]
    allowed_tools: tuple[str, ...]
    context_refs: tuple[str, ...]
    prompt: str
    sha256: str


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=1200)
    kind: Literal["observation", "hypothesis", "knowledge"]
    evidence_refs: list[str] = Field(
        default_factory=list,
        max_length=12,
        description='Exact retained handles, e.g. ["E17", "E18"]. '
        "Repeat the E prefix on every entry; never use bare numbers.",
    )
    counterevidence_refs: list[str] = Field(
        default_factory=list,
        max_length=12,
        description='Exact contrary evidence handles, e.g. ["E2", "E3"].',
    )
    knowledge_refs: list[str] = Field(default_factory=list, max_length=8)


class Hypothesis(Finding):
    id: str = Field(min_length=1, max_length=30)
    assessment: Literal["open", "supported", "refuted"] = "open"
    missing_observations: list[str] = Field(default_factory=list, max_length=4)


class InvestigationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[Finding] = Field(max_length=6)
    missing_observations: list[str] = Field(max_length=5)
    outcome: Literal["answered", "partial", "blocked"] = "answered"
    answered_criteria: list[str] = Field(default_factory=list, max_length=4)
    reason_code: Literal[
        "missing_data", "permission", "budget", "context_limit",
        "tool_failure",
        "model_failure", "invalid_report",
    ] | None = None


class TaskRecord(BaseModel):
    id: str
    round: int
    task: InvestigationTask
    status: Literal[
        "pending", "running", "completed", "incomplete", "failed"
    ] = "pending"
    report: InvestigationReport | None = None
    input_revision: str = ""
    inputs: TaskInput | None = None
    report_sha256: str | None = None
    tool_limit: int = Field(ge=0)
    tool_calls: int = Field(default=0, ge=0)
    token_limit: int | None = Field(default=None, ge=0)
    tokens: TokenLedger = Field(default_factory=TokenLedger)


class CollaborationState(BaseModel):
    version: Literal["graph-v7"] = "graph-v7"
    phase: Literal["coordinate", "investigate", "review", "completed"] = (
        "coordinate"
    )
    round: int = Field(default=0, ge=0, le=2)
    decision: int = Field(default=0, ge=0, le=6)
    batches: int = Field(default=0, ge=0, le=3)
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=6)
    tasks: dict[str, TaskRecord] = Field(default_factory=dict)
    active_tasks: list[str] = Field(default_factory=list, max_length=2)
    feedback: list[Claim] = Field(default_factory=list)
    repairs: dict[str, int] = Field(default_factory=dict)
    draft: Diagnosis | None = None
    reviews: list[VerificationRecord] = Field(default_factory=list)
    tool_results: dict[str, list[dict]] = Field(default_factory=dict)
    tool_result_bytes: int = 0
