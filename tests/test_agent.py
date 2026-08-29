import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
import json
from uuid import uuid4

import httpx
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
import pytest
from tool_fixtures import FixtureToolClient

from tracemind.agent.errors import DiagnosisError
from tracemind.agent.runtime.harness import RunHarness
from tracemind.agent.runtime.llm import create_chat_model
from tracemind.agent.runtime.runner import LangChainAgentRunner
from tracemind.agent.service import DiagnosisService
from tracemind.config import Settings
from tracemind.models import (
    DiagnosisStatus,
    Evidence,
    EvidenceSource,
    Incident,
)
from tracemind.models.result import DiagnosisDraft
from tracemind.models.run import RunLimits, RunState


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
        EvidenceUsingRunner(),
        tool_client=FixtureToolClient(
            StubPrometheus(), UnusedJaeger(), UnusedOpenSearch()
        ),
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
        HallucinatingRunner(),
        tool_client=FixtureToolClient(
            StubPrometheus(), UnusedJaeger(), UnusedOpenSearch()
        ),
    )

    with pytest.raises(DiagnosisError, match="no telemetry tool returned"):
        asyncio.run(
            service.diagnose(
                Incident(title="failure", description="unknown"),
                start=datetime(2024, 9, 1, tzinfo=UTC),
                end=datetime(2024, 9, 1, 1, tzinfo=UTC),
            )
        )


def test_model_factory_supports_google_openai_and_deepseek() -> None:
    google = create_chat_model(Settings(google_api_key="google-test-key"))
    openai = create_chat_model(
        Settings(
            llm_provider="openai",
            llm_model="gpt-5.6-luna",
            openai_api_key="openai-test-key",
        )
    )
    deepseek = create_chat_model(
        Settings(
            _env_file=None,
            llm_provider="deepseek",
            llm_model="deepseek-flash",
            deepseek_api_key="deepseek-test-key",
        )
    )

    assert isinstance(google, ChatGoogleGenerativeAI)
    assert isinstance(openai, ChatOpenAI)
    assert isinstance(deepseek, BaseChatOpenAI)
    assert all(model.max_retries == 0 for model in (google, openai, deepseek))
    assert str(deepseek.openai_api_base) == "https://api.deepseek.com"
    assert deepseek.use_responses_api is False
    assert deepseek.extra_body == {"thinking": {"type": "disabled"}}
    assert deepseek.profile["structured_output"] is False


def test_model_factory_requires_selected_provider_key() -> None:
    with pytest.raises(ValueError, match="TRACEMIND_GOOGLE_API_KEY"):
        create_chat_model(Settings(google_api_key=""))
    with pytest.raises(ValueError, match="TRACEMIND_DEEPSEEK_API_KEY"):
        create_chat_model(Settings(
            _env_file=None, llm_provider="deepseek", deepseek_api_key="",
            openai_api_key="unrelated-key",
        ))


def test_deepseek_agent_sends_compatible_structured_tool_request(monkeypatch):
    """Exercise the agent and SDK against a simulated DeepSeek endpoint."""
    def respond(request):
        assert str(request.url) == "https://api.deepseek.com/chat/completions"
        assert request.headers["authorization"] == "Bearer deepseek-test-key"
        payload = json.loads(request.content)
        assert payload["max_tokens"] == 512
        assert "max_completion_tokens" not in payload
        assert payload["thinking"] == {"type": "disabled"}
        assert "response_format" not in payload
        assert any(tool["function"]["name"] == "DiagnosisDraft"
                   for tool in payload["tools"])
        return httpx.Response(200, json={
            "id": "deepseek-test", "object": "chat.completion", "created": 0,
            "model": "deepseek-flash",
            "choices": [{"index": 0, "finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": "",
                                     "tool_calls": [{
                                         "id": "draft-1", "type": "function",
                                         "function": {
                                             "name": "DiagnosisDraft",
                                             "arguments": json.dumps({
                                                 "status": "inconclusive",
                                                 "root_cause": "No evidence",
                                                 "confidence": 0,
                                                 "evidence_ids": [],
                                                 "remediation": [],
                                             }),
                                         },
                                     }]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20,
                      "total_tokens": 30},
        })

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond)
        ) as http:
            monkeypatch.setattr(
                "tracemind.agent.runtime.llm.BaseChatOpenAI",
                lambda **kwargs: BaseChatOpenAI(
                    http_async_client=http, **kwargs
                ),
            )
            model = create_chat_model(Settings(
                _env_file=None, llm_provider="deepseek",
                llm_model="deepseek-flash",
                deepseek_api_key="deepseek-test-key",
            ))
            state = RunState(
                incident=Incident(title="Failure", description="HTTP 500"),
                investigation_start=datetime(2024, 9, 1, tzinfo=UTC),
                investigation_end=datetime(2024, 9, 1, 1, tzinfo=UTC),
                limits=RunLimits(output_tokens=512), allowed_tools=(),
            )
            draft = await LangChainAgentRunner(model).run(
                "Investigate", [], recursion_limit=8,
                harness=RunHarness(state),
            )
            assert draft.status is DiagnosisStatus.INCONCLUSIVE
            assert state.usage.reported_total_tokens == 30

    asyncio.run(run())
