"""Fault fixtures and unscored experiment outputs."""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, Field

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


class ExperimentOutput(BaseModel):
    """Observed task output, with no independent scoring or report format."""

    kind: Literal["diagnosis", "retrieval"] = "diagnosis"
    diagnosis: Diagnosis | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    run_state: RunState | None = None
    error_type: str | None = None
    elapsed_ms: float = Field(default=0, ge=0)
    model_provider: str = ""
    model_name: str = ""
    execution_mode: str = "live"
    investigation_start: AwareDatetime | None = None
    investigation_end: AwareDatetime | None = None
    memory_mode: Literal["off", "frozen"] = "off"
    memory_corpus_id: str | None = None
    telemetry_transport: str = "mcp"
    retrieval: dict[str, Any] | None = None
    knowledge: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_saved(cls, saved: dict) -> "ExperimentOutput":
        """Project saved diagnostics, excluding scores and model histories."""
        state = saved.get("run_state") or saved.get("state") or {}
        return cls(
            diagnosis=saved.get("diagnosis"),
            evidence=saved.get("evidence", []),
            run_state=state or None,
            error_type=saved.get("error_type"),
            investigation_start=(
                saved.get("investigation_start")
                or state.get("investigation_start")
            ),
            investigation_end=(
                saved.get("investigation_end")
                or state.get("investigation_end")
            ),
            elapsed_ms=saved.get("elapsed_ms", 0),
            model_provider=saved.get("model_provider", "unknown"),
            model_name=saved.get("model_name", "unknown"),
            execution_mode="saved",
            knowledge=saved.get("knowledge") or state.get("knowledge", []),
        )
