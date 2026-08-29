"""Durable local worker for queued and interrupted diagnosis runs."""

import asyncio
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
import logging
import os

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from tracemind.agent.errors import (
    DiagnosisCancelled,
    DiagnosisError,
    ModelConfigurationError,
)
from tracemind.agent.factory import build_runner, build_service
from tracemind.agent.storage.io import run_io
from tracemind.agent.storage.memory_repository import MemoryRepository
from tracemind.agent.storage.run_repository import RunRepository
from tracemind.agent.service import DiagnosisService
from tracemind.config import Settings
from tracemind.mcp.client import HttpMcpClient, open_telemetry_backend
from tracemind.models.run import RunStatus, StopReason
from tracemind.models.storage import ClaimedRun, RunConflict


logger = logging.getLogger(__name__)


class RunWorker:
    """Poll SQLite, lease one run, and resume its LangGraph thread."""

    def __init__(
        self, settings: Settings, *, repository: RunRepository | None = None,
        service_factory: Callable[[AsyncSqliteSaver, HttpMcpClient],
                                  DiagnosisService] | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self._service_factory = service_factory

    def _service(
        self, saver: AsyncSqliteSaver, tool_client: HttpMcpClient,
    ) -> DiagnosisService:
        if self._service_factory is not None:
            return self._service_factory(saver, tool_client)
        memory = MemoryRepository(self.repository.database)
        return build_service(
            self.settings, build_runner(self.settings, checkpointer=saver),
            tool_client,
            memory_policy=self.settings.memory_policy,
            memory_lookup=memory.snapshot,
            guidance_lookup=memory.guidance_snapshot,
        )

    async def _monitor(self, claim: ClaimedRun, task: asyncio.Task) -> None:
        """Renew ownership and deliver cancellation to the active task."""
        while not task.done():
            await asyncio.sleep(1)
            if task.done():
                break
            try:
                cancel = await run_io(
                    self.repository.heartbeat,
                    claim.run.state.id, claim.owner
                )
            except RunConflict:
                cancel = True
            if cancel:
                task.cancel()
                break

    async def run_once(self) -> bool:
        """Execute one claim, finalizing cancellation before worker shutdown.

        Cancelling a diagnosis leaves the worker available for another claim.
        Cancelling the worker task propagates after resources and state close.
        Return False when the queue is empty.
        """
        if self.repository is None:
            self.repository = await run_io(
                RunRepository, self.settings.agent_run_database)
        worker_task = asyncio.current_task()
        # Claiming transfers ownership. Retain its result even if shutdown
        # arrives while SQLite is committing, then finalize the claimed run.
        claiming = asyncio.create_task(run_io(self.repository.claim_next))
        while not claiming.done():
            try:
                await asyncio.shield(claiming)
            except asyncio.CancelledError:
                continue
        claim = claiming.result()
        if claim is None:
            if worker_task.cancelling():
                raise asyncio.CancelledError
            return False
        state = claim.run.state
        evidence = claim.run.evidence
        run_id = state.id
        try:
            if worker_task.cancelling():
                raise asyncio.CancelledError
            checkpoint_path = self.settings.agent_checkpoint_database
            os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
            checkpoint_path.parent.mkdir(
                parents=True,
                exist_ok=True,
                mode=0o700)
            try:
                descriptor = os.open(
                    checkpoint_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600,
                )
            except FileExistsError:
                pass
            else:
                os.close(descriptor)

            async with AsyncSqliteSaver.from_conn_string(
                str(checkpoint_path)
            ) as saver:
                checkpoint = await saver.aget_tuple({
                    "configurable": {"thread_id": str(run_id)}
                })
                async with (
                    open_telemetry_backend(self.settings) as tool_client
                ):
                    service = self._service(saver, tool_client)
                    task = asyncio.create_task(service.run_persisted(
                        state, evidence,
                        persist=lambda current, items: (
                            self.repository.save_progress(
                                run_id, claim.owner, current, items
                            )
                        ),
                        is_cancelled=lambda: self.repository.should_stop(
                            run_id, claim.owner
                        ),
                        resume=claim.resuming,
                        graph_resume=claim.resuming and checkpoint is not None,
                        feedback_writer=lambda current: MemoryRepository(
                            self.repository.database).apply_feedback(
                                current, owner=claim.owner),
                    ))
                    monitor = asyncio.create_task(self._monitor(claim, task))
                    try:
                        run = await task
                        state, evidence = run.state or state, run.evidence
                    except DiagnosisError as exc:
                        state = exc.state or state
                        evidence = exc.evidence
                    finally:
                        monitor.cancel()
                        with suppress(asyncio.CancelledError):
                            await monitor
        except asyncio.CancelledError as exc:
            if isinstance(exc, DiagnosisCancelled):
                state, evidence = exc.state, exc.evidence
            state.status = RunStatus.CANCELLED
            state.stop_reason = StopReason.CANCELLED
        except RunConflict:
            logger.warning("Run %s: worker lease lost", run_id)
            if worker_task.cancelling():
                raise asyncio.CancelledError
            return True
        except Exception as exc:
            logger.error("Run %s: worker failed (%s)",
                         run_id, type(exc).__name__)
            state.status = RunStatus.FAILED
            state.stop_reason = (
                StopReason.MODEL_ERROR
                if isinstance(exc, ModelConfigurationError)
                else StopReason.INTERNAL_ERROR
            )

        if state.status not in {
            RunStatus.SUCCEEDED, RunStatus.INCONCLUSIVE,
            RunStatus.FAILED, RunStatus.CANCELLED,
        }:
            state.status = RunStatus.FAILED
            state.stop_reason = StopReason.INTERNAL_ERROR
        if state.finished_at is None:
            state.finished_at = datetime.now(UTC)
        try:
            await run_io(
                self.repository.finish, run_id, claim.owner, state, evidence)
        except RunConflict:
            logger.warning("Run %s: worker lease lost before finalization",
                           run_id)
        else:
            logger.info("Run %s finished: %s", run_id, state.status.value)
        # A cancelled child diagnosis does not cancel this task. A SIGINT or
        # caller cancellation does, and must stop polling after finalization.
        if worker_task.cancelling():
            raise asyncio.CancelledError
        return True

    async def run_forever(self, *, poll_seconds: float = 1) -> None:
        """Keep claiming work, including leases left by dead processes."""
        while True:
            if not await self.run_once():
                await asyncio.sleep(poll_seconds)
