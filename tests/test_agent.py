import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from tracemind.agent import DiagnosisError, DiagnosisService, create_chat_model
from tracemind.agent.service import DiagnosisDraft
from tracemind.config import Settings
from tracemind.models import (
    DiagnosisStatus,
    Evidence,
    EvidenceSource,
    Incident,
)


class StubPrometheus:
    async def query(self, expression, *, at=None, limit=50):
        return [
            Evidence(
                source=EvidenceSource.METRIC,
                summary="Metric request_errors_total=42",
                observed_at=at,
                service="checkout",
                query=expression,
            )
        ]


class UnusedJaeger:
    async def search_traces(self, service, *, start, end, limit=20):
        raise AssertionError("trace tool should not be called")


class UnusedOpenSearch:
    async def search_logs(self, service, *, start, end, text=None, limit=100):
        raise AssertionError("log tool should not be called")


class EvidenceUsingRunner:
    async def run(
        self,
        prompt: str,
        tools: Sequence[BaseTool],
        *,
        recursion_limit: int,
        harness,
    ) -> DiagnosisDraft:
        assert "checkout errors" in prompt
        assert recursion_limit == 8
        metrics_tool = next(
            tool for tool in tools if tool.name == "query_metrics"
        )
        evidence = await metrics_tool.ainvoke(
            {"expression": "sum(request_errors_total)"}
        )
        return DiagnosisDraft(
            status=DiagnosisStatus.COMPLETE,
            root_cause="Checkout error rate increased",
            confidence=0.85,
            evidence_ids=[evidence[0]["id"]],
            remediation=["Inspect the checkout deployment"],
        )


class HallucinatingRunner:
    async def run(self, prompt, tools, *, recursion_limit, harness):
        return DiagnosisDraft(
            status=DiagnosisStatus.COMPLETE,
            root_cause="Unsupported conclusion",
            confidence=0.9,
            evidence_ids=[uuid4()],
        )


def test_service_returns_diagnosis_with_tool_evidence() -> None:
    incident = Incident(title="checkout errors", description="HTTP 500 spike")
    service = DiagnosisService(
        StubPrometheus(),
        UnusedJaeger(),
        UnusedOpenSearch(),
        EvidenceUsingRunner(),
        recursion_limit=8,
    )

    diagnosis = asyncio.run(
        service.diagnose(
            incident,
            start=datetime(2024, 9, 1, tzinfo=UTC),
            end=datetime(2024, 9, 1, 1, tzinfo=UTC),
        )
    )

    assert diagnosis.incident_id == incident.id
    assert diagnosis.status is DiagnosisStatus.COMPLETE
    assert len(diagnosis.evidence_ids) == 1


def test_service_rejects_evidence_id_not_returned_by_tool() -> None:
    service = DiagnosisService(
        StubPrometheus(),
        UnusedJaeger(),
        UnusedOpenSearch(),
        HallucinatingRunner(),
    )

    with pytest.raises(DiagnosisError, match="no telemetry tool returned"):
        asyncio.run(
            service.diagnose(
                Incident(title="failure", description="unknown"),
                start=datetime(2024, 9, 1, tzinfo=UTC),
                end=datetime(2024, 9, 1, 1, tzinfo=UTC),
            )
        )


def test_model_factory_supports_google_and_openai() -> None:
    google = create_chat_model(Settings(google_api_key="google-test-key"))
    openai = create_chat_model(
        Settings(
            llm_provider="openai",
            llm_model="gpt-5.6-luna",
            openai_api_key="openai-test-key",
        )
    )

    assert isinstance(google, ChatGoogleGenerativeAI)
    assert isinstance(openai, ChatOpenAI)
    assert google.max_retries == openai.max_retries == 0


def test_model_factory_requires_selected_provider_key() -> None:
    with pytest.raises(ValueError, match="TRACEMIND_GOOGLE_API_KEY"):
        create_chat_model(Settings(google_api_key=""))
