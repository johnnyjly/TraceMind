from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tracemind.models import (
    Diagnosis,
    DiagnosisStatus,
    Evidence,
    EvidenceSource,
    Incident,
)


def test_evidence_links_to_a_diagnosis() -> None:
    incident = Incident(
        title="Checkout latency increased",
        description="P95 latency increased from 300 ms to 2.5 s.",
        affected_services=["checkout"],
    )
    evidence = Evidence(
        source=EvidenceSource.METRIC,
        summary="Checkout P95 latency is above the expected range.",
        observed_at=datetime.now(UTC),
        service="checkout",
        query="histogram_quantile(...)",
    )

    diagnosis = Diagnosis(
        incident_id=incident.id,
        status=DiagnosisStatus.COMPLETE,
        root_cause="Payment dependency latency",
        confidence=0.8,
        evidence_ids=[evidence.id],
        remediation=["Inspect the payment service dependency."],
    )

    assert diagnosis.incident_id == incident.id
    assert diagnosis.evidence_ids == [evidence.id]


def test_diagnosis_rejects_confidence_outside_unit_interval() -> None:
    incident = Incident(title="Failure", description="An incident occurred.")

    with pytest.raises(ValidationError):
        Diagnosis(
            incident_id=incident.id,
            status=DiagnosisStatus.INCONCLUSIVE,
            root_cause="Unknown",
            confidence=1.1,
        )
