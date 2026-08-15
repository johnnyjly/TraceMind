"""Exercise the real LangGraph loop through the persistent local worker."""

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import Field

from tracemind.agent.persistence import RunRepository
from tracemind.agent.errors import DiagnosisCancelled
from tracemind.agent.harness import RunHarness
from tracemind.agent.middleware import HarnessMiddleware
from tracemind.agent.service import (
    SYSTEM_PROMPT, DiagnosisDraft, DiagnosisService, LangChainAgentRunner,
)
from tracemind.agent.tools import EvidenceCollector, create_telemetry_tools
from tracemind.agent.worker import RunWorker
from tracemind.config import Settings
from tracemind.models import Evidence, EvidenceSource, Incident
from tracemind.models.run import RunLimits, RunState, RunStatus


START = datetime(2024, 9, 1, tzinfo=UTC)


class ScriptedModel(BaseChatModel):
    """Issue one tool call, then cite its returned evidence."""

    calls: int = 0
    bindings: list[list[str]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-persistence-test"

    def bind_tools(self, tools, **kwargs):
        self.bindings.append([
            tool.name if hasattr(tool, "name")
            else tool.get("function", tool)["name"] for tool in tools
        ])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("the worker must use the async model path")

    async def _agenerate(
        self, messages, stop=None, run_manager=None, **kwargs
    ):
        self.calls += 1
        tool_messages = [
            item for item in messages
            if isinstance(item, ToolMessage) and item.name == "query_metrics"
        ]
        if not tool_messages:
            response = AIMessage(content="", tool_calls=[{
                "name": "query_metrics",
                "args": {"expression": "up"},
                "id": "query-1",
            }])
        else:
            evidence_id = json.loads(tool_messages[-1].content)[0]["id"]
            response = AIMessage(content="", tool_calls=[{
                "name": "DiagnosisDraft",
                "args": {
                    "status": "complete", "root_cause": "Payment failed",
                    "confidence": 0.8, "evidence_ids": [evidence_id],
                    "remediation": [],
                },
                "id": "diagnosis-1",
            }])
        response.usage_metadata = {
            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
        }
        return ChatResult(generations=[ChatGeneration(message=response)])


class Telemetry:
    """Return one stable metric observation."""

    def __init__(self) -> None:
        self.calls = 0

    async def query(self, expression, *, at=None):
        self.calls += 1
        return [Evidence(
            source=EvidenceSource.METRIC,
            summary="payment failures=10", service="payment",
            observed_at=START,
        )]


def test_worker_persists_graph_and_terminal_result(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        agent_run_database=tmp_path / "runs.sqlite3",
        agent_checkpoint_database=tmp_path / "checkpoints.sqlite3",
        agent_trace_directory=tmp_path / "traces",
    )
    repository = RunRepository(settings.agent_run_database)
    state = RunState(
        incident=Incident(title="checkout error", description="HTTP 500"),
        investigation_start=START,
        investigation_end=START + timedelta(hours=1),
        limits=RunLimits(), allowed_tools=("query_metrics",),
        status=RunStatus.QUEUED,
    )
    repository.create(state, request_hash="worker-test")
    model = ScriptedModel()
    telemetry = Telemetry()

    def service_factory(saver, _http_client):
        return DiagnosisService(
            telemetry, telemetry, telemetry,
            LangChainAgentRunner(model, checkpointer=saver),
            event_directory=settings.agent_trace_directory,
        )

    async def execute():
        worker = RunWorker(
            settings, repository=repository, service_factory=service_factory
        )
        assert await worker.run_once()
        async with AsyncSqliteSaver.from_conn_string(
            str(settings.agent_checkpoint_database)
        ) as saver:
            checkpoint = await saver.aget_tuple({
                "configurable": {"thread_id": str(state.id)}
            })
        return checkpoint

    checkpoint = asyncio.run(execute())
    stored = RunRepository(settings.agent_run_database).get(state.id)
    assert checkpoint is not None
    assert stored.state.status is RunStatus.SUCCEEDED
    assert stored.state.model_calls == model.calls == 2
    assert stored.state.tool_calls == {"query_metrics": 1}
    assert len(stored.evidence) == 1
    assert stored.result is not None
    assert stored.result.diagnosis.evidence_ids == [stored.evidence[0].id]


def test_resume_reuses_checkpoint_and_retained_evidence(
    tmp_path: Path,
) -> None:
    """Restart after a tool step without reissuing its telemetry query."""
    settings = Settings(
        _env_file=None,
        agent_run_database=tmp_path / "runs.sqlite3",
        agent_checkpoint_database=tmp_path / "checkpoints.sqlite3",
        agent_trace_directory=tmp_path / "traces",
    )
    repository = RunRepository(settings.agent_run_database)
    state = RunState(
        incident=Incident(title="checkout error", description="HTTP 500"),
        investigation_start=START,
        investigation_end=START + timedelta(hours=1),
        limits=RunLimits(), allowed_tools=("query_metrics",),
        status=RunStatus.QUEUED,
    )
    repository.create(state, request_hash="restart-test")
    claim = repository.claim_next()
    assert claim is not None
    telemetry = Telemetry()
    first_model = ScriptedModel()

    async def first_step() -> None:
        collector: EvidenceCollector
        collector = EvidenceCollector(
            on_change=lambda: repository.save_progress(
                state.id, claim.owner, claim.run.state, collector.evidence
            ),
        )
        harness = RunHarness(
            claim.run.state,
            event_directory=settings.agent_trace_directory,
            persist=lambda current: repository.save_progress(
                state.id, claim.owner, current, collector.evidence
            ),
        )
        tools = create_telemetry_tools(
            telemetry, telemetry, telemetry, start=START,
            end=START + timedelta(hours=1), collector=collector,
            harness=harness,
        )
        async with AsyncSqliteSaver.from_conn_string(
            str(settings.agent_checkpoint_database)
        ) as saver:
            agent = create_agent(
                first_model, tools, system_prompt=SYSTEM_PROMPT,
                response_format=DiagnosisDraft,
                middleware=[HarnessMiddleware(harness)],
                checkpointer=saver, interrupt_after=["tools"],
            )
            await agent.ainvoke(
                {"messages": [{"role": "user", "content": "Investigate"}]},
                {"configurable": {"thread_id": str(state.id)}},
            )
        harness._close_trace()

    asyncio.run(first_step())
    halfway = repository.get(state.id)
    assert halfway.state.model_calls == 1
    assert halfway.state.tool_calls == {"query_metrics": 1}
    assert len(halfway.evidence) == telemetry.calls == 1
    with sqlite3.connect(settings.agent_run_database) as connection:
        connection.execute(
            "UPDATE runs SET lease_until=0 WHERE id=?", (str(state.id),)
        )

    resumed_model = ScriptedModel()

    def service_factory(saver, _http_client):
        return DiagnosisService(
            telemetry, telemetry, telemetry,
            LangChainAgentRunner(resumed_model, checkpointer=saver),
            event_directory=settings.agent_trace_directory,
        )

    worker = RunWorker(
        settings, repository=RunRepository(settings.agent_run_database),
        service_factory=service_factory,
    )
    assert asyncio.run(worker.run_once())
    completed = repository.get(state.id)
    assert completed.state.status is RunStatus.SUCCEEDED
    assert completed.state.model_calls == 2
    assert completed.state.tool_calls == {"query_metrics": 1}
    assert telemetry.calls == 1
    assert completed.result is not None
    assert completed.result.diagnosis.evidence_ids == [halfway.evidence[0].id]


def test_active_cancellation_stops_worker(tmp_path: Path) -> None:
    """A cancellation request interrupts an active claim and finalizes it."""
    settings = Settings(
        _env_file=None,
        agent_run_database=tmp_path / "runs.sqlite3",
        agent_checkpoint_database=tmp_path / "checkpoints.sqlite3",
    )
    repository = RunRepository(settings.agent_run_database)
    state = RunState(
        incident=Incident(title="checkout error", description="HTTP 500"),
        investigation_start=START,
        investigation_end=START + timedelta(hours=1),
        limits=RunLimits(), allowed_tools=("query_metrics",),
        status=RunStatus.QUEUED,
    )
    repository.create(state, request_hash="cancel-test")
    entered = asyncio.Event()

    class WaitingService:
        """Represent a model call waiting when cancellation arrives."""

        async def run_persisted(self, *_args, **_kwargs):
            state, evidence = _args
            evidence.append(Evidence(
                source=EvidenceSource.METRIC,
                summary="retained before cancellation", observed_at=START,
            ))
            _kwargs["persist"](state, evidence)
            entered.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError as exc:
                raise DiagnosisCancelled(state, evidence) from exc

    async def execute() -> None:
        worker = RunWorker(
            settings, repository=repository,
            service_factory=lambda _saver, _client: WaitingService(),
        )
        task = asyncio.create_task(worker.run_once())
        await entered.wait()
        repository.cancel(state.id)
        assert await asyncio.wait_for(task, timeout=3)

    asyncio.run(execute())
    stored = repository.get(state.id)
    assert stored.state.status is RunStatus.CANCELLED
    assert stored.cancel_requested
    assert stored.state.stop_reason.value == "cancelled"
    assert len(stored.evidence) == 1
    assert repository.claim_next() is None
