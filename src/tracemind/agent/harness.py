"""Per-run execution budgets, lifecycle controls, and metadata events."""

import asyncio
import hashlib
import json
import logging
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any, NoReturn, TypeVar
from uuid import UUID, uuid4

from tracemind.agent.errors import DiagnosisError
from tracemind.agent.llm import MODEL_SERVICE_ERRORS
from tracemind.models.run import RunEvent, RunState, RunStatus, StopReason
from tracemind.telemetry.errors import TelemetryClientError


T = TypeVar("T")
logger = logging.getLogger(__name__)


class RunStopped(DiagnosisError):
    """Signal a harness-enforced stop with its machine-readable reason."""

    def __init__(self, reason: StopReason) -> None:
        """Attach the stop reason to an investigation control-flow
        exception.
        """
        super().__init__(f"Diagnosis stopped: {reason.value}")
        self.reason = reason


class RunHarness:
    """Enforce per-run admission, budgets, deadlines, and lifecycle events.

    One harness belongs to one asyncio event loop. Call reservations happen
    before awaiting work, so concurrent tools share a budget. The first
    stop reason is latched; the service drains tools and finalizes the run.
    JSONL events are best-effort diagnostics, not recovery checkpoints.
    """

    def __init__(
        self, state: RunState, *, event_directory: Path | None = None
    ) -> None:
        """Start a run timer and optionally open an exclusive JSONL trace
        file.
        """
        self.state = state
        self._started = monotonic()
        self._stop: StopReason | None = None
        self._signatures: dict[str, int] = {}
        self._active: set[asyncio.Task[Any]] = set()
        self._model_span: UUID | None = None
        self._stream = None
        if event_directory is not None:
            try:
                event_directory.mkdir(parents=True, exist_ok=True)
                fd = os.open(
                    event_directory / f"{state.id}.jsonl",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                self._stream = os.fdopen(fd, "w", encoding="utf-8")
            except OSError as exc:
                logger.warning(
                    "Run %s: cannot open trace file (%s); "
                    "keeping events in memory",
                    state.id, type(exc).__name__,
                )
        self.emit("run.started", span_id=state.id, data={
            "prompt_version": state.prompt_version,
            "allowed_tools": list(state.allowed_tools),
            **state.limits.model_dump(),
        })

    @property
    def remaining_seconds(self) -> float:
        """Return the nonnegative run time remaining on the monotonic
        clock.
        """
        return max(
            0,
            self.state.limits.duration_seconds - (monotonic() - self._started),
        )

    def emit(
        self, kind: str, *, span_id: UUID | None = None,
        parent_span_id: UUID | None = None, duration_ms: float | None = None,
        data: dict | None = None,
    ) -> None:
        """Append ordered operational metadata and attempt a JSONL write.

        Callers must supply sanitized metadata; this method does not redact
        it. File I/O errors disable the file sink while retaining in-memory
        events.
        """
        event = RunEvent(
            run_id=self.state.id, sequence=len(self.state.events) + 1,
            kind=kind,
            span_id=span_id or uuid4(), parent_span_id=parent_span_id,
            duration_ms=duration_ms, data=data or {},
        )
        self.state.events.append(event)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "%s", event.model_dump_json(),
                extra={"run_id": str(event.run_id),
                       "span_id": str(event.span_id),
                       "event_kind": event.kind},
            )
        if self._stream is not None:
            operation = "write"
            try:
                self._stream.write(event.model_dump_json() + "\n")
                operation = "flush"
                self._stream.flush()
            except OSError as exc:
                logger.warning(
                    "Run %s: trace %s failed (%s); keeping events in memory",
                    self.state.id, operation, type(exc).__name__,
                )
                self._close_trace()

    def _close_trace(self) -> None:
        """Detach and close the trace stream, logging close failures."""
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.close()
            except OSError as exc:
                logger.warning("Run %s: trace close failed (%s)",
                               self.state.id, type(exc).__name__)

    def stop(self, reason: StopReason) -> NoReturn:
        """Latch the first stop reason and raise RunStopped on every
        invocation.

        This blocks further admission; it does not finalize state or cancel
        tasks. The service owns draining and terminal status selection.
        """
        if self._stop is None:
            self._stop = reason
            self.emit("policy.stopped", parent_span_id=self.state.id,
                      data={"reason": reason.value})
        raise RunStopped(self._stop)

    def check_active(self) -> None:
        """Raise RunStopped if stopped, finalized, or past the run deadline."""
        if self._stop is not None:
            raise RunStopped(self._stop)
        if self.state.status is not RunStatus.RUNNING:
            raise RunStopped(
                self.state.stop_reason or StopReason.INTERNAL_ERROR
            )
        if self.remaining_seconds <= 0:
            self.stop(StopReason.DEADLINE)

    def authorize_tool(self, name: str) -> None:
        """Check run activity and the trusted tool-name allowlist.

        Unknown or hidden names stop the run before tool execution. This
        check does not implement service-level or tenant-level
        authorization.
        """
        self.check_active()
        if name not in self.state.allowed_tools:
            self.stop(StopReason.TOOL_DENIED)

    async def model_call(
        self, call: Callable[[], Awaitable[T]], *, model_name: str
    ) -> T:
        """Reserve one model call, enforce its timeout, and record
        usage/events.

        Failed admitted calls consume budget. Missing provider usage is
        recorded as unknown. Expected provider errors and timeouts stop the
        run; cancellation and unexpected exceptions propagate. No retries
        are made.
        """
        self.check_active()
        if self.state.model_calls >= self.state.limits.model_calls:
            self.stop(StopReason.MODEL_BUDGET)
        self.state.model_calls += 1
        span = uuid4()
        self._model_span = span
        started = monotonic()
        self.emit("model.started", span_id=span, parent_span_id=self.state.id,
                  data={"model": model_name, "call": self.state.model_calls})
        reported = False
        call_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        try:
            async with asyncio.timeout(
                self.state.limits.model_timeout_seconds
            ):
                response = await call()
            self.check_active()
            # ModelResponse contains the provider AIMessage,
            # including usage if supplied.
            usage = [getattr(item, "usage_metadata", None)
                     for item in getattr(response, "result", ())]
            for item in usage:
                if item is not None:
                    reported = True
                    for key in call_usage:
                        call_usage[key] += item.get(key, 0)
                    self.state.usage.reported_input_tokens += item.get(
                        "input_tokens", 0
                    )
                    self.state.usage.reported_output_tokens += item.get(
                        "output_tokens", 0
                    )
                    self.state.usage.reported_total_tokens += item.get(
                        "total_tokens", 0
                    )
            self.emit("model.completed",
                      span_id=span, parent_span_id=self.state.id,
                      duration_ms=(monotonic() - started) * 1000,
                      data={"usage_reported": reported,
                            **{key: value if reported else None
                               for key, value in call_usage.items()}})
            return response
        except TimeoutError:
            self.emit("model.failed",
                      span_id=span, parent_span_id=self.state.id,
                      duration_ms=(monotonic() - started) * 1000,
                      data={"error_type": "TimeoutError"})
            self.stop(StopReason.MODEL_TIMEOUT)
        except asyncio.CancelledError:
            self.emit("model.cancelled",
                      span_id=span, parent_span_id=self.state.id,
                      duration_ms=(monotonic() - started) * 1000)
            raise
        except Exception as exc:
            self.emit("model.failed",
                      span_id=span, parent_span_id=self.state.id,
                      duration_ms=(monotonic() - started) * 1000,
                      data={"error_type": type(exc).__name__})
            if isinstance(exc, MODEL_SERVICE_ERRORS):
                self.stop(StopReason.MODEL_ERROR)
            raise
        finally:
            if reported:
                self.state.usage.reported_model_calls += 1
            else:
                self.state.usage.unreported_model_calls += 1

    async def tool_call(
        self, name: str, arguments: dict[str, object],
        call: Callable[[], Awaitable[list[dict[str, object]]]],
    ) -> list[dict[str, object]]:
        """Authorize and reserve a tool call before awaiting its result.

        Count admitted calls, including failures, against shared and
        identical-argument limits. The callback stores evidence and selects
        bounded results. Record returned IDs and timing; telemetry errors
        and timeouts stop the run. Other errors and cancellation propagate.
        """
        self.authorize_tool(name)
        if sum(self.state.tool_calls.values()) >= self.state.limits.tool_calls:
            self.stop(StopReason.TOOL_BUDGET)
        signature = hashlib.sha256(json.dumps(
            [name, arguments], sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if (
            self._signatures.get(signature, 0)
            >= self.state.limits.identical_tool_calls
        ):
            self.stop(StopReason.REPEATED_TOOL_CALL)
        # No await before these reservations:
        # parallel tool calls share one budget.
        self._signatures[signature] = self._signatures.get(signature, 0) + 1
        self.state.tool_calls[name] = self.state.tool_calls.get(name, 0) + 1
        span = uuid4()
        parent = self._model_span or self.state.id
        started = monotonic()
        task = asyncio.current_task()
        assert task is not None
        self._active.add(task)
        self.emit("tool.started", span_id=span, parent_span_id=parent,
                  data={"tool": name, "arguments_sha256": signature})
        try:
            async with asyncio.timeout(self.state.limits.tool_timeout_seconds):
                result = await call()
            self.check_active()
            self.emit("tool.completed", span_id=span, parent_span_id=parent,
                      duration_ms=(monotonic() - started) * 1000,
                      data={"tool": name, "result_count": len(result),
                            "evidence_ids": [
                                str(item["id"]) for item in result
                                if "id" in item
                            ]})
            return result
        except TimeoutError:
            self.emit("tool.failed", span_id=span, parent_span_id=parent,
                      duration_ms=(monotonic() - started) * 1000,
                      data={"tool": name, "error_type": "TimeoutError"})
            self.stop(StopReason.TOOL_TIMEOUT)
        except asyncio.CancelledError:
            self.emit("tool.cancelled", span_id=span, parent_span_id=parent,
                      duration_ms=(monotonic() - started) * 1000,
                      data={"tool": name})
            raise
        except Exception as exc:
            self.emit("tool.failed", span_id=span, parent_span_id=parent,
                      duration_ms=(monotonic() - started) * 1000,
                      data={"tool": name, "error_type": type(exc).__name__})
            if isinstance(exc, TelemetryClientError):
                self.stop(StopReason.TOOL_ERROR)
            raise
        finally:
            self._active.discard(task)

    async def drain(self) -> None:
        """Cancel sibling tool tasks before producing a terminal snapshot."""
        tasks = [
            task for task in self._active
            if task is not asyncio.current_task()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def finish(self, status: RunStatus, reason: StopReason) -> None:
        """Finalize a running state, emit its terminal event, and close the
        sink.

        Repeated calls leave terminal state unchanged. Callers must drain
        active tools first; this method does not cancel or await them.
        """
        if self.state.status is not RunStatus.RUNNING:
            return
        self.state.status = status
        self.state.stop_reason = reason
        self.state.finished_at = datetime.now(UTC)
        self.state.elapsed_ms = (monotonic() - self._started) * 1000
        try:
            self.emit("run.finished", span_id=self.state.id,
                      duration_ms=self.state.elapsed_ms,
                      data={"status": status.value, "reason": reason.value,
                            "model_calls": self.state.model_calls,
                            "tool_calls": sum(self.state.tool_calls.values()),
                            **self.state.usage.model_dump()})
        finally:
            self._close_trace()
