"""Exercise the real LangGraph loop through the persistent local worker."""

from proposal_fixtures import proposal_payload

import asyncio
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import Field
import pytest
from tool_fixtures import FixtureToolClient, bind_fixture_tools

from tracemind.agent.errors import DiagnosisCancelled
from tracemind.agent.runtime.harness import RunHarness
from tracemind.agent.storage.io import run_io
from tracemind.agent.runtime.middleware import HarnessMiddleware
from tracemind.agent.storage.run_repository import RunRepository
from tracemind.agent.runtime.runner import LangChainAgentRunner, SYSTEM_PROMPT
from tracemind.agent.service import DiagnosisService
from tracemind.agent.runtime.tools import EvidenceCollector
from tracemind.agent.worker import RunWorker
from tracemind.config import Settings
from tracemind.models import Evidence, EvidenceSource, Incident
from tracemind.models.result import DiagnosisDraft
from tracemind.models.run import RunLimits, RunState, RunStatus, StopReason


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
                "name": "DiagnosisProposal",
                "args": proposal_payload({
                    "status": "complete", "root_cause": "Payment failed",
                    "confidence": 0.8, "evidence_ids": [evidence_id],
                    "remediation": [],
                }),
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
            LangChainAgentRunner(model, checkpointer=saver),
            tool_client=FixtureToolClient(telemetry, telemetry, telemetry),
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


@pytest.mark.parametrize("finalizing", [False, True])
@pytest.mark.parametrize("model_budget", [2, 8])
def test_resume_reuses_checkpoint_and_retained_evidence(
    tmp_path: Path, finalizing: bool, model_budget: int,
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
        limits=RunLimits(model_calls=model_budget),
        allowed_tools=("query_metrics",),
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
            persist=lambda current: run_io(
                repository.save_progress,
                state.id, claim.owner, current, collector.evidence
            ),
        )
        tools = bind_fixture_tools(
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
        if finalizing:
            harness.begin_finalization(StopReason.NO_PROGRESS)
        await harness.checkpoint(check_cancel=False)
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
            LangChainAgentRunner(resumed_model, checkpointer=saver),
            tool_client=FixtureToolClient(telemetry, telemetry, telemetry),
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
    if finalizing or model_budget == 2:
        assert resumed_model.bindings == [["DiagnosisProposal"]]
        phases = [e for e in completed.state.events
                  if e.kind == "policy.finalizing"]
        assert len(phases) == 1
        assert phases[0].data["reason"] == (
            "no_progress" if finalizing else "model_budget")


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


@pytest.mark.parametrize("forever", [False, True])
@pytest.mark.parametrize("phase", ["checkpoint", "model"])
def test_worker_shutdown_finalizes_claim_and_preserves_queued_work(
    tmp_path: Path, monkeypatch, forever, phase,
) -> None:
    """Worker cancellation cleans up and exits without claiming another run."""
    settings = Settings(
        _env_file=None, agent_run_database=tmp_path / "runs.sqlite3",
        agent_checkpoint_database=tmp_path / "checkpoints.sqlite3",
        agent_trace_directory=tmp_path / "traces",
    )
    repository = RunRepository(settings.agent_run_database)
    first = RunState(
        incident=Incident(title="checkout failure", description="HTTP 500"),
        investigation_start=START,
        investigation_end=START + timedelta(hours=1),
        limits=RunLimits(), allowed_tools=("query_metrics",),
        status=RunStatus.QUEUED,
    )
    second = first.model_copy(deep=True, update={"id": uuid4()})
    repository.create(first, request_hash="active-shutdown")
    repository.create(second, request_hash="keep-queued")
    entered = asyncio.Event()

    class WaitingModel(ScriptedModel):
        """Block the second model call after real evidence was persisted."""

        async def _agenerate(self, messages, **kwargs):
            if self.calls == 1:
                self.calls += 1
                entered.set()
                await asyncio.Future()
            return await super()._agenerate(messages, **kwargs)

    if phase == "checkpoint":
        async def wait_for_checkpoint(*_args, **_kwargs):
            """Interrupt setup after the run is claimed, before diagnosis."""
            entered.set()
            await asyncio.Future()

        monkeypatch.setattr(
            AsyncSqliteSaver, "aget_tuple", wait_for_checkpoint,
        )
    model, telemetry = WaitingModel(), Telemetry()

    async def execute():
        worker = RunWorker(settings, repository=repository,
                           service_factory=lambda saver, _client:
                           DiagnosisService(
                               LangChainAgentRunner(model, checkpointer=saver),
                               tool_client=FixtureToolClient(
                                   telemetry, telemetry, telemetry
                               ),
                               event_directory=settings.agent_trace_directory,
                           ))
        task = asyncio.create_task(
            worker.run_forever() if forever else worker.run_once()
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=3)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(execute())
    saved = RunRepository(repository.path).get(first.id)
    assert saved.state.status is RunStatus.CANCELLED
    assert saved.state.stop_reason is StopReason.CANCELLED
    assert saved.state.finished_at is not None
    # Worker stop is distinct from an API cancellation request.
    assert not saved.cancel_requested
    assert len(saved.evidence) == (1 if phase == "model" else 0)
    assert saved.state.model_calls == (2 if phase == "model" else 0)
    assert saved.state.tool_calls == (
        {"query_metrics": 1} if phase == "model" else {}
    )
    terminal = [e for e in saved.state.events if e.kind == "run.finished"]
    assert len(terminal) == 1
    assert terminal[0].data["status"] == "cancelled"
    assert repository.get(second.id).state.status is RunStatus.QUEUED
    with sqlite3.connect(repository.path) as connection:
        assert connection.execute(
            "SELECT lease_owner,lease_until FROM runs WHERE id=?",
            (str(first.id),),
        ).fetchone() == (None, None)


def test_api_cancellation_keeps_worker_running_for_next_run(tmp_path):
    """Cancelling only the diagnosis still permits a queued run to complete."""
    settings = Settings(
        _env_file=None, agent_run_database=tmp_path / "runs.sqlite3",
        agent_checkpoint_database=tmp_path / "checkpoints.sqlite3",
        agent_trace_directory=tmp_path / "traces",
    )
    repository = RunRepository(settings.agent_run_database)
    first = RunState(
        incident=Incident(title="checkout failure", description="HTTP 500"),
        investigation_start=START,
        investigation_end=START + timedelta(hours=1),
        limits=RunLimits(), allowed_tools=("query_metrics",),
        status=RunStatus.QUEUED,
    )
    second = first.model_copy(deep=True, update={"id": uuid4()})
    repository.create(first, request_hash="api-cancel")
    repository.create(second, request_hash="after-api-cancel")
    entered = asyncio.Event()

    class WaitingModel(ScriptedModel):
        """Wait only in the first run; the next run uses normal responses."""

        async def _agenerate(self, messages, **kwargs):
            if self.calls == 1:
                self.calls += 1
                entered.set()
                await asyncio.Future()
            return await super()._agenerate(messages, **kwargs)

    model, telemetry = WaitingModel(), Telemetry()

    async def execute():
        worker = RunWorker(settings, repository=repository,
                           service_factory=lambda saver, _client:
                           DiagnosisService(
                               LangChainAgentRunner(model, checkpointer=saver),
                               tool_client=FixtureToolClient(
                                   telemetry, telemetry, telemetry
                               ),
                               event_directory=settings.agent_trace_directory,
                           ))
        task = asyncio.create_task(worker.run_forever())

        async def wait_for_result():
            """Wait for the second run's durable terminal snapshot."""
            while repository.get(second.id).state.status is not (
                RunStatus.SUCCEEDED
            ):
                await asyncio.sleep(.01)

        try:
            await asyncio.wait_for(entered.wait(), timeout=3)
            repository.cancel(first.id)
            await asyncio.wait_for(wait_for_result(), timeout=3)
            assert not task.done()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(execute())
    cancelled = repository.get(first.id)
    assert cancelled.state.status is RunStatus.CANCELLED
    assert cancelled.cancel_requested
    assert len(cancelled.evidence) == 1
    assert repository.get(second.id).result.diagnosis.evidence_ids
