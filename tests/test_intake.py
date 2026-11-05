"""Initial hints are bounded, sanitized and distinct from observations."""

import hashlib
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tracemind.api.app import request_fingerprint
from tracemind.api.schemas import DiagnosisRequest, FollowUpRequest
from tracemind.models import Incident
from tracemind.models.intake import InitialContext
from tracemind.models.run import RunLimits, RunState


CONTEXT = {
    "origin": {"source": "alert", "external_id": "alert-42",
               "reported_at": "2024-09-01T00:01:00Z"},
    "traces": [{"trace_id": "AB" * 16, "span_id": "CD" * 8,
                "service": "checkout", "operation": "POST /checkout"}],
    "error": {"type": "TimeoutError", "message": "upstream timed out",
              "stacktrace": "checkout.py:42\n  await charge()",
              "file": "checkout.py", "line": 42},
    "scope": {"environment": "production", "cluster": "west",
              "namespace": "store"},
    "deployments": [{"service": "checkout", "version": "v2",
                     "instance": "checkout-1"}],
}


def test_context_round_trip_normalizes_ids_without_creating_evidence():
    context = InitialContext.model_validate(CONTEXT)
    assert context.traces[0].trace_id == "ab" * 16
    assert context.traces[0].span_id == "cd" * 8
    state = RunState(
        incident=Incident(title="checkout", description="timeouts"),
        initial_context=context,
        investigation_start=datetime(2024, 9, 1, tzinfo=UTC),
        investigation_end=datetime(2024, 9, 1, 1, tzinfo=UTC),
        limits=RunLimits(), allowed_tools=(),
    )
    restored = RunState.model_validate_json(state.model_dump_json())
    assert restored.initial_context == context
    assert restored.evidence_ids == []
    assert "initial_context" not in Incident.model_fields
    legacy = state.model_dump(exclude={"initial_context"})
    assert RunState.model_validate(legacy).initial_context is None


@pytest.mark.parametrize("payload", [
    {"traces": [{"trace_id": "a" * 15}]},
    {"traces": [{"trace_id": "a" * 20}]},
    {"traces": [{"trace_id": "0" * 32}]},
    {"traces": [{"trace_id": "g" * 32}]},
    {"traces": [{"trace_id": "a" * 32, "span_id": "0" * 16}]},
    {"traces": [{"trace_id": "a" * 32, "span_id": "b" * 32}]},
    {"traces": [{"span_id": "b" * 16}]},
    {"traces": [{"trace_id": "a" * 32}] * 4},
    {"traces": [{"trace_id": "a" * 32, "unknown": "field"}]},
    {"deployments": [{"service": "checkout"}] * 11},
    {"deployments": [{"version": "v2"}]},
    {"error": {"message": "a" * 4097}},
    {"error": {"stacktrace": "a" * 16385}},
    {"error": {"stacktrace": "错" * 12000}},
    {"error": {"line": 0}},
    {"error": {"line": "42"}},
    {"origin": {"reported_at": "2024-09-01T00:01:00"}},
    {"scope": {"environment": "  "}},
    {"evidence_ids": ["invented-evidence"]},
])
def test_context_rejects_invalid_or_unbounded_hints(payload):
    with pytest.raises(ValidationError):
        InitialContext.model_validate(payload)


def test_context_accepts_64_bit_trace_without_span():
    context = InitialContext(traces=[{"trace_id": " A" + "B" * 15 + " "}])
    assert context.traces[0].trace_id == "a" + "b" * 15
    assert context.traces[0].span_id is None


def test_context_redacts_recognizable_credentials_before_serialization():
    context = InitialContext(error={
        "message": 'password=secret-one token=secret-two '
        'https://user:secret-three@host/path?api_key=secret-four '
        '"client_secret": "secret-five"',
        "stacktrace": "Authorization: Bearer secret-six\n"
        "Cookie: session=secret-seven; other=secret-eight\n"
        "at checkout.py:42\nBearer secret-nine\n"
        "-----BEGIN PRIVATE KEY-----\nsecret-ten\n"
        "-----END PRIVATE KEY-----",
    })
    serialized = context.model_dump_json()
    assert "secret-" not in serialized
    assert "[REDACTED]" in serialized
    assert "checkout.py:42" in serialized
    assert InitialContext.model_validate_json(serialized) == context


@pytest.mark.parametrize("payload", [
    DiagnosisRequest(
        incident=Incident(title="checkout", description="HTTP 500"),
        investigation_start="2024-09-01T00:00:00Z",
        investigation_end="2024-09-01T01:00:00Z",
    ),
    FollowUpRequest(message="explain", purpose="explain"),
])
def test_absent_context_preserves_historical_fingerprint(payload):
    excluded = {"initial_context"}
    legacy = payload.model_dump(mode="json", exclude=excluded)
    if isinstance(payload, DiagnosisRequest):
        legacy["incident"].pop("id")
    canonical = json.dumps(legacy, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode()).hexdigest()
    assert request_fingerprint(payload) == expected


def test_context_participates_in_fingerprint_after_id_normalization():
    request = FollowUpRequest(message="investigate", initial_context=CONTEXT)
    normalized = request.model_copy(deep=True)
    normalized.initial_context.traces[0].trace_id = "ab" * 16
    assert request_fingerprint(request) == request_fingerprint(normalized)
    changed = request.model_copy(deep=True)
    changed.initial_context.deployments[0].version = "v3"
    assert request_fingerprint(request) != request_fingerprint(changed)


def test_explanation_rejects_fresh_context():
    with pytest.raises(ValidationError, match="cannot accept"):
        FollowUpRequest(message="why", purpose="explain", initial_context={})
