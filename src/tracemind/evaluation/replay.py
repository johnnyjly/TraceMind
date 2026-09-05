"""Fixed telemetry replay through the real agent graph, without network I/O."""

import json
from pathlib import Path
from typing import Literal

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from mcp.types import CallToolResult, Tool
from pydantic import AwareDatetime, BaseModel, Field, model_validator

from tracemind.agent.memory.lookup import evaluation_memory
from tracemind.agent.runtime.runner import LangChainAgentRunner
from tracemind.agent.service import DiagnosisService
from tracemind.config import Settings
from tracemind.evaluation.models import EvaluationCase, EvaluationReport
from tracemind.evaluation.runner import run_service_evaluation
from tracemind.evaluation.summary import trace_issues
from tracemind.mcp.contracts import TelemetryReply
from tracemind.models import Evidence
from tracemind.models.memory import MemorySnapshot
from tracemind.models.run import StopReason
from tracemind.telemetry.errors import TelemetryClientError


class ReplayResponse(BaseModel):
    """One expected backend request and its observations or safe failure."""

    method: Literal[
        "query", "search_traces", "get_trace_detail", "search_logs",
    ]
    arguments: dict[str, str | int | float | bool | None]
    evidence: list[Evidence] = Field(default_factory=list)
    error: Literal["transient", "unavailable"] | None = None
    metadata: dict | None = None

    @model_validator(mode="after")
    def validate_payload(self):
        if self.metadata is not None and (self.evidence or self.error):
            raise ValueError("metadata cannot accompany evidence or error")
        return self


class ReplayFixture(BaseModel):
    """Telemetry transcript and separate scripted model actions.

    The script is a harness regression oracle, not an accuracy benchmark.
    A real model receives only telemetry; it never receives the script or case.
    """

    case: EvaluationCase
    start: AwareDatetime
    end: AwareDatetime
    responses: list[ReplayResponse]
    script: list[dict]
    expected_stop_reason: StopReason | None = None

    @model_validator(mode="after")
    def validate_window(self) -> "ReplayFixture":
        """Reject empty, reversed or timezone-free replay windows."""
        if self.end <= self.start:
            raise ValueError("replay end must be after start")
        return self

    @classmethod
    def load(cls, path: Path) -> "ReplayFixture":
        """Validate a committed synthetic replay fixture."""
        return cls.model_validate_json(path.read_text())


class ReplayTelemetry:
    """Consume exact backend requests and reject unrecorded calls."""

    transport = "replay"

    def __init__(self, responses: list[ReplayResponse]) -> None:
        """Copy the transcript so concurrent runs consume independent lists."""
        self.responses = list(responses)
        self.calls = 0

    async def _respond(
        self, method: str, arguments: dict,
    ) -> list[Evidence] | dict:
        """Consume the first matching response, preserving retry order."""
        for index, response in enumerate(self.responses):
            if response.method == method and response.arguments == arguments:
                self.responses.pop(index)
                self.calls += 1
                if response.error == "transient":
                    try:
                        raise httpx.ConnectError("synthetic transport failure")
                    except httpx.ConnectError as exc:
                        raise TelemetryClientError(
                            "replay unavailable"
                        ) from exc
                if response.error == "unavailable":
                    raise TelemetryClientError("replay unavailable")
                return (response.metadata if response.metadata is not None
                        else response.evidence)
        raise AssertionError(f"unrecorded replay request: {method}")

    async def discover_tools(self) -> list[Tool]:
        return recorded_tools()

    async def call(self, name: str, arguments: dict) -> CallToolResult:
        arguments = {k: v for k, v in arguments.items()
                     if k not in {"start", "end", "at"}}
        if name == "search_logs":
            arguments.setdefault("text", None)
        result = await self._respond(
            "query" if name == "query_metrics" else name, arguments)
        reply = (TelemetryReply(metadata=result) if isinstance(result, dict)
                 else TelemetryReply(evidence=result))
        return CallToolResult(content=[], structuredContent=reply.model_dump(
            mode="json", exclude_none=True))


def recorded_tools() -> list[Tool]:
    """Load versioned wire definitions for reproducible offline replays."""
    path = Path(__file__).with_name("telemetry_tools.json")
    return [Tool.model_validate(item) for item in json.loads(path.read_text())]


class ScriptedReplayModel(BaseChatModel):
    """Emit fixed actions; substitute observed citation IDs at runtime."""

    script: list[dict]
    position: int = 0

    @property
    def _llm_type(self) -> str:
        """Identify scripted measurements in the harness model events."""
        return "scripted-replay"

    def bind_tools(self, tools, **kwargs):
        """Use the tool schema supplied by the real agent graph."""
        return self

    def _generate(self, messages, **kwargs):
        """Reject synchronous invocation of the async replay model."""
        raise AssertionError("replay requires async execution")

    async def _agenerate(self, messages, **kwargs):
        """Build one tool or diagnosis message without invented token usage."""
        if self.position >= len(self.script):
            raise AssertionError("script exhausted")
        action = dict(self.script[self.position])
        self.position += 1
        if action["name"] == "DiagnosisDraft":
            arguments = dict(action["args"])
            indices = arguments.pop("cite_indices", [])
            observed = []
            for message in messages:
                if isinstance(message, ToolMessage):
                    try:
                        payload = json.loads(message.content)
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if isinstance(payload, list):
                        for item in payload:
                            if isinstance(item, dict) and "id" in item:
                                if item["id"] not in observed:
                                    observed.append(item["id"])
            arguments["evidence_ids"] = [observed[i] for i in indices]
            action["args"] = arguments
        response = AIMessage(content="", tool_calls=[{
            **action, "id": f"replay-{self.position}",
        }])
        return ChatResult(generations=[ChatGeneration(message=response)])


async def run_replay_evaluation(
    fixture: ReplayFixture, settings: Settings, *,
    model: BaseChatModel | None = None,
    memory_snapshot: MemorySnapshot | None = None,
) -> EvaluationReport:
    """Run scripted regression or an explicitly supplied real model offline."""
    policy, lookup = evaluation_memory(
        memory_snapshot, settings.memory_policy, start=fixture.start,
        max_records=settings.agent_memory_max_records,
        context_bytes=settings.agent_memory_context_bytes,
    )
    telemetry = ReplayTelemetry(fixture.responses)
    selected_model = (model if model is not None else
                      ScriptedReplayModel(script=fixture.script))
    limits = settings.run_limits
    limits.tool_calls = min(limits.tool_calls, fixture.case.max_tool_calls)
    service = DiagnosisService(
        LangChainAgentRunner(selected_model),
        tool_client=telemetry,
        limits=limits,
        allowed_tools=settings.agent_allowed_tools,
        recursion_limit=settings.agent_recursion_limit,
        event_directory=settings.agent_trace_directory,
        memory_policy=policy,
        memory_lookup=lookup,
    )
    report = await run_service_evaluation(
        fixture.case, service, start=fixture.start, end=fixture.end,
        model_provider=settings.llm_provider if model else "scripted",
        model_name=settings.llm_model if model else "scripted-replay",
        execution_mode="model_replay" if model else "scripted_replay",
        memory_corpus_id=(memory_snapshot.fingerprint
                          if memory_snapshot else None),
    )
    outcome = (
        report.run_state is not None and
        report.run_state.stop_reason is fixture.expected_stop_reason
        if fixture.expected_stop_reason is not None else
        bool(report.result and report.result.passed)
    )
    report.regression_passed = bool(
        outcome and report.run_state is not None
        and not trace_issues(report.run_state)
    )
    if model is None and (telemetry.responses or
                          selected_model.position != len(fixture.script)):
        report.regression_passed = False
    return report
