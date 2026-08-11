import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from tracemind.models import Diagnosis, DiagnosisStatus, Evidence, EvidenceSource, Incident
from tracemind.models.run import RunState


class FaultInjection(BaseModel):
    feature_flag: str
    variant: str


class GroundTruth(BaseModel):
    root_cause_service: str = Field(min_length=1)
    root_cause_terms: list[str] = Field(min_length=1)
    required_evidence_source: EvidenceSource


class EvaluationCase(BaseModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    incident: Incident
    fault: FaultInjection
    ground_truth: GroundTruth
    max_tool_calls: int = Field(default=12, ge=1)

    @classmethod
    def load(cls, path: Path) -> "EvaluationCase":
        return cls.model_validate(json.loads(path.read_text()))


class EvaluationResult(BaseModel):
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


class EvaluationReport(BaseModel):
    case_id: str
    model_provider: str
    model_name: str
    investigation_start: datetime
    investigation_end: datetime
    diagnosis: Diagnosis
    evidence: list[Evidence]
    result: EvaluationResult
    run_state: RunState | None = None
