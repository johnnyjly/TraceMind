"""Fault scenarios, expected outcomes, and evaluation report schemas."""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from tracemind.models import (
    Diagnosis,
    DiagnosisStatus,
    Evidence,
    EvidenceSource,
    Incident,
)
from tracemind.models.run import RunState


class FaultInjection(BaseModel):
    """Demo feature flag and variant defining a fault injection."""

    feature_flag: str
    variant: str


class GroundTruth(BaseModel):
    """Held-out expected status and checks for complete root-cause answers."""

    root_cause_service: str = Field(min_length=1)
    root_cause_terms: list[str] = Field(min_length=1)
    required_evidence_source: EvidenceSource
    expected_status: DiagnosisStatus = DiagnosisStatus.COMPLETE


class EvaluationCase(BaseModel):
    """Incident fixture with optional fault, ground truth, and tool
    budget.
    """

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    incident: Incident
    fault: FaultInjection | None = None
    ground_truth: GroundTruth
    max_tool_calls: int = Field(default=12, ge=1)

    @classmethod
    def load(cls, path: Path) -> "EvaluationCase":
        """Read and validate a JSON case; propagate file and validation
        errors.
        """
        return cls.model_validate(json.loads(path.read_text()))


class EvaluationResult(BaseModel):
    """Individual scoring checks, measured duration, and aggregate pass
    status.
    """

    case_id: str
    diagnosis_status: DiagnosisStatus
    root_cause_service_match: bool
    root_cause_term_match: bool
    required_evidence_match: bool
    evidence_grounded: bool
    tool_call_count: int
    within_tool_budget: bool
    elapsed_ms: float = Field(ge=0)
    passed: bool
    false_completion: bool = False


class EvaluationReport(BaseModel):
    """Evaluation result with model identity, investigation data, and run
    state.
    """

    case_id: str
    model_provider: str
    model_name: str
    investigation_start: datetime
    investigation_end: datetime
    diagnosis: Diagnosis | None = None
    evidence: list[Evidence]
    result: EvaluationResult | None = None
    run_state: RunState | None = None
    error_type: str | None = None
    execution_mode: str = "live"
    elapsed_ms: float = Field(default=0, ge=0)
    regression_passed: bool | None = None
    memory_mode: Literal["off", "frozen"] = "off"
    memory_corpus_id: str | None = None
