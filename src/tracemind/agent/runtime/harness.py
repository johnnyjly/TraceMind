"""Per-run execution budgets, lifecycle controls, and metadata events."""

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import random
from time import monotonic
from typing import Any, NoReturn, TypeVar
from uuid import UUID, uuid4

from tracemind.agent.errors import DiagnosisError
from tracemind.agent.runtime.llm import MODEL_SERVICE_ERRORS
from tracemind.agent.storage.io import run_io
from tracemind.agent.workflow.tool_policy import resolve_tool_policy
from tracemind.agent.runtime.recovery import (
    Failure,
    ModelCallFailed,
    ToolUnavailable,
    classify_failure,
)
from tracemind.models.run import RunEvent, RunState, RunStatus, StopReason
from tracemind.telemetry.errors import TelemetryClientError


T = TypeVar("T")
logger = logging.getLogger(__name__)


def _cache_usage(message) -> tuple[int, int] | None:
    """Read normalized usage or DeepSeek's original chat-completion fields."""
    usage = getattr(message, "usage_metadata", None) or {}
    hit = (usage.get("input_token_details") or {}).get("cache_read")
    total = usage.get("input_tokens")
    if hit is None:
        metadata = getattr(message, "response_metadata", {})
        raw = metadata.get("token_usage") or {}
        hit = raw.get("prompt_cache_hit_tokens")
        total = raw.get("prompt_tokens", total)
    # Missing or inconsistent external usage must not look like a zero hit.
    if type(hit) is not int or type(total) is not int or not 0 <= hit <= total:
        return None
    return hit, total - hit


class RunStopped(DiagnosisError):
    """Signal a harness-enforced stop with its machine-readable reason."""

    def __init__(self, reason: StopReason) -> None:
        """Attach the stop reason to an investigation control-flow
        exception.
        """
        super().__init__(f"Diagnosis stopped: {reason.value}")
        self.reason = reason


class InvestigationClosed(Exception):
    """Reject further tool admission while allowing a final diagnosis."""


class TaskToolBudgetExhausted(Exception):
    """End only the executing task's tool allocation, not the run."""

    def __init__(self, task_id: str):
        super().__init__("Task tool allocation exhausted")
        self.task_id = task_id


class RunHarness:
    """Enforce per-run admission, budgets, deadlines, and lifecycle events.

    One harness belongs to one asyncio event loop. Call reservations happen
    before awaiting work, so concurrent tools share a budget. The first
    stop reason is latched; the service drains tools and finalizes the run.
    JSONL events are best-effort diagnostics, not recovery checkpoints.
    """

    def __init__(
        self,
        state: RunState,
        *,
        event_directory: Path | None = None,
        persist: Callable[[RunState], Awaitable[None]] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
        resume: bool = False,
        graph_resume: bool | None = None,
    ) -> None:
        """Start a run timer and optionally open an exclusive JSONL trace
        file.
        """
        self.evidence = lambda: []
        self.state = state
        state.allowed_tools = resolve_tool_policy(
            state.allowed_tools,
            purpose=state.purpose,
        )
        self._started = monotonic()
        self._persist = persist
        self._is_cancelled = is_cancelled
        self._persistence_lock = asyncio.Lock()
        self.resuming = resume
        self.graph_resuming = resume if graph_resume is None else graph_resume
        self._role = ContextVar("agent_role", default=None)
        self._task_id = ContextVar("agent_task_id", default=None)
        self._inflight_tools = {}
        self.verifier = None
        self._stop: StopReason | None = None
        self._signatures = dict(state.tool_signatures)
        self._active: set[asyncio.Task[Any]] = set()
        self._model_spans: dict[str | None, UUID] = {}
        self._tool_span: ContextVar[UUID | None] = ContextVar(
            "tool_span",
            default=None,
        )
        self.validate_draft: Callable[[Any], None] | None = None
        self._result_fingerprints = set(state.result_fingerprints)
        self._progress = state.progress_count
        self._last_progress = state.last_progress
        # The durable phase event survives a restart independently of the
        # graph checkpoint, including a pending tool-node replay.
        self.finalization_reason = next(
            (
                StopReason(event.data["reason"])
                for event in reversed(state.events)
                if event.kind == "policy.finalizing"
            ),
            None,
        )
        self._stream = None
        if event_directory is not None:
            try:
                event_directory.mkdir(parents=True, exist_ok=True)
                fd = os.open(
                    event_directory / f"{state.id}.jsonl",
                    (
                        os.O_WRONLY | os.O_CREAT | os.O_APPEND
                        if resume
                        else os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    ),
                    0o600,
                )
                self._stream = os.fdopen(fd, "w", encoding="utf-8")
            except OSError as exc:
                logger.warning(
                    "Run %s: cannot open trace file (%s); "
                    "keeping events in memory",
                    state.id,
                    type(exc).__name__,
                )
        if not resume:
            self.emit(
                "run.started",
                span_id=state.id,
                data={
                    "prompt_version": state.prompt_version,
                    "telemetry_transport": state.telemetry_transport,
                    "allowed_tools": list(state.allowed_tools),
                    **state.limits.model_dump(),
                },
            )
        else:
            self.emit(
                "run.resumed",
                span_id=state.id,
                data={
                    "telemetry_transport": state.telemetry_transport,
                },
            )
            started = {
                event.span_id: event
                for event in state.events
                if event.kind in {"model.started", "tool.started"}
            }
            closed = {
                event.span_id
                for event in state.events
                if event.kind
                in {
                    "model.completed",
                    "model.failed",
                    "model.cancelled",
                    "tool.completed",
                    "tool.failed",
                    "tool.cancelled",
                }
            }
            for span_id, event in started.items():
                if span_id not in closed:
                    self.emit(
                        "recovery.ambiguous_call",
                        span_id=span_id,
                        parent_span_id=state.id,
                        data={"call_kind": event.kind},
                    )

    @property
    def role(self):
        return self._role.get()

    @role.setter
    def role(self, value):
        self._role.set(value)

    async def shared_tool_call(self, name, arguments, call, **kwargs):
        """Coalesce identical run-local reads; only completed results persist.

        The key includes the fixed window, snapshot and tool policy. A crash
        after a remote response but before persistence may repeat a read; its
        previous admission remains charged. Cached reads add no backend call.
        """
        workflow = self.state.collaboration
        if workflow is None or name in {"read_evidence", "read_knowledge"}:
            return await self.tool_call(name, arguments, call, **kwargs)
        self.authorize_tool(name)
        key = hashlib.sha256(
            json.dumps(
                {
                    "tool": name,
                    "arguments": arguments,
                    "start": str(self.state.investigation_start),
                    "end": str(self.state.investigation_end),
                    "snapshot": (
                        self.state.knowledge_policy.snapshot
                        if self.state.knowledge_policy
                        else None
                    ),
                    "permissions": self.state.allowed_tools,
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        if key in workflow.tool_results:
            self.emit("tool.reused", data={"tool": name, "key": key})
            return deepcopy(workflow.tool_results[key])
        if key in self._inflight_tools:
            result = await asyncio.shield(self._inflight_tools[key])
            self.check_active()
            self.emit("tool.reused", data={"tool": name, "key": key})
            return deepcopy(result)
        future = asyncio.get_running_loop().create_future()
        self._inflight_tools[key] = future
        try:
            result = await self.tool_call(name, arguments, call, **kwargs)
            size = len(json.dumps(result, default=str).encode())
            if workflow.tool_result_bytes + size <= 262144:
                workflow.tool_results[key] = deepcopy(result)
                workflow.tool_result_bytes += size
            await self.checkpoint()
            future.set_result(deepcopy(result))
            return result
        except BaseException as exc:
            future.set_exception(exc)
            # The owner also observes failures without waiters.
            future.exception()
            raise
        finally:
            self._inflight_tools.pop(key, None)

    async def checkpoint(self, *, check_cancel: bool = True) -> None:
        """Commit reservations/results before external work or graph return.

        Snapshot under one lock so concurrent tools cannot persist older state
        after newer state. Admission counters are reserved before awaiting.
        """
        async with self._persistence_lock:
            if check_cancel and self._is_cancelled is not None:
                if await run_io(self._is_cancelled):
                    self.stop(StopReason.CANCELLED)
            if self._persist is not None:
                await self._persist(self.state.model_copy(deep=True))
        if check_cancel:
            self.check_active()

    @property
    def remaining_seconds(self) -> float:
        """Return the nonnegative run time remaining on the monotonic
        clock.
        """
        if self.state.deadline_at is not None:
            return max(
                0, (self.state.deadline_at - datetime.now(UTC)).total_seconds()
            )
        return max(
            0,
            self.state.limits.duration_seconds - (monotonic() - self._started),
        )

    def emit(
        self,
        kind: str,
        *,
        span_id: UUID | None = None,
        parent_span_id: UUID | None = None,
        duration_ms: float | None = None,
        data: dict | None = None,
    ) -> None:
        """Append metadata or bounded diagnosis excerpts and write JSONL.

        Callers must supply sanitized metadata; this method does not redact
        it. File I/O errors disable the file sink while retaining in-memory
        events.
        """
        event = RunEvent(
            run_id=self.state.id,
            sequence=len(self.state.events) + 1,
            kind=kind,
            span_id=span_id or uuid4(),
            parent_span_id=parent_span_id,
            duration_ms=duration_ms,
            data={
                **({"role": self.role} if self.role else {}),
                **(
                    {"task_id": self._task_id.get()}
                    if self._task_id.get()
                    else {}
                ),
                **(data or {}),
            },
        )
        self.state.events.append(event)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "%s",
                event.model_dump_json(),
                extra={
                    "run_id": str(event.run_id),
                    "span_id": str(event.span_id),
                    "event_kind": event.kind,
                },
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
                    self.state.id,
                    operation,
                    type(exc).__name__,
                )
                self._close_trace()

    def _close_trace(self) -> None:
        """Detach and close the trace stream, logging close failures."""
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.close()
            except OSError as exc:
                logger.warning(
                    "Run %s: trace close failed (%s)",
                    self.state.id,
                    type(exc).__name__,
                )

    def stop(self, reason: StopReason) -> NoReturn:
        """Latch the first stop reason and raise RunStopped on every
        invocation.

        This blocks further admission; it does not finalize state or cancel
        tasks. The service owns draining and terminal status selection.
        """
        if self._stop is None:
            self._stop = reason
            self.emit(
                "policy.stopped",
                parent_span_id=self.state.id,
                data={"reason": reason.value},
            )
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
        if self.finalization_reason is not None:
            raise InvestigationClosed()

    def begin_finalization(self, reason: StopReason) -> None:
        """Close tool admission once, without extending any hard budget.

        Admitted tools can still return observations. The next model request
        receives the completed tool exchanges and submits a diagnosis.
        """
        self.check_active()
        if self.state.model_calls >= (
            self.state.limits.model_calls - self.verification_calls
        ):
            self.stop(StopReason.MODEL_BUDGET)
        if self.finalization_reason is None:
            self.finalization_reason = reason
            self.emit(
                "policy.finalizing",
                parent_span_id=self.state.id,
                data={"reason": reason.value},
            )

    @property
    def verification_calls(self) -> int:
        return 2 * int(
            self.state.verification_policy is not None
            and self.state.verification is None
        )

    @property
    def finalization_calls(self) -> int:
        """Reserve synthesis and remaining repair, leaving a discovery turn."""
        limits = self.state.limits
        workflow = self.state.collaboration
        used = (
            workflow.repairs.get(self._task_id.get(), 0)
            if workflow
            else self.state.diagnosis_repairs
        )
        return (
            min(
                1 + max(0, limits.diagnosis_repairs - used),
                max(1, limits.model_calls - 1 - self.verification_calls),
            )
            + self.verification_calls
        )

    @property
    def auxiliary_reserve(self):
        workflow = self.state.collaboration
        if workflow is None:
            return self.finalization_calls
        pending = sum(
            workflow.tasks[key].status in {"pending", "running"}
            for key in workflow.active_tasks
        )
        return max(
            self.finalization_calls,
            pending
            + 1
            + self.state.limits.diagnosis_repairs
            + self.verification_calls,
        )

    async def model_call(
        self,
        call: Callable[[], Awaitable[T]],
        *,
        model_name: str,
        allow_fallback: bool = False,
        purpose: str = "diagnosis",
        required: bool = True,
        reserved_calls: int = 0,
        before_attempt: Callable[[], None] | None = None,
        retry_attempts: int | None = None,
    ) -> T:
        """Retry transient calls within the shared call budget and deadline.

        Optional calls may return a recoverable failure to their workflow;
        run admission and hard limits apply equally to every call.
        """
        retries = (
            self.state.limits.retry_attempts
            if retry_attempts is None
            else retry_attempts
        )
        for attempt in range(retries + 1):
            try:
                if before_attempt is not None:
                    before_attempt()
                return await self._model_attempt(
                    call,
                    model_name=model_name,
                    purpose=purpose,
                    reserved_calls=reserved_calls,
                )
            except (TimeoutError, *MODEL_SERVICE_ERRORS) as exc:
                failure = classify_failure(exc)
                if failure.retryable and attempt < retries:
                    await self._wait_retry("model", attempt, failure)
                    continue
                timed_out = isinstance(exc, TimeoutError)
                if not required or (allow_fallback and failure.retryable):
                    raise ModelCallFailed(
                        failure, timed_out=timed_out
                    ) from exc
                self.stop(
                    StopReason.MODEL_TIMEOUT
                    if timed_out
                    else StopReason.MODEL_ERROR
                )
        raise AssertionError("retry loop must return or raise")

    async def _wait_retry(
        self,
        target: str,
        attempt: int,
        failure: Failure,
    ) -> None:
        """Schedule cancellable backoff, honoring server hints and deadline."""
        self.check_active()
        limits = self.state.limits
        delay = min(
            limits.retry_max_backoff_seconds,
            limits.retry_backoff_seconds * 2**attempt,
        )
        delay *= random.uniform(0.5, 1)
        delay = max(delay, failure.retry_after or 0)
        if delay >= self.remaining_seconds:
            self.stop(StopReason.DEADLINE)
        self.emit(
            "recovery.retry_scheduled",
            parent_span_id=self.state.id,
            data={
                "target": target,
                "retry": attempt + 1,
                "delay_seconds": delay,
                "category": failure.category,
                "status_code": failure.status_code,
            },
        )
        await self.checkpoint()
        await asyncio.sleep(delay)
        self.check_active()

    def check_progress(self) -> None:
        """Close investigation after rounds without new usable observations."""
        if self.finalization_reason is not None:
            return
        if self._last_progress is not None:
            self.state.stalled_rounds = (
                self.state.stalled_rounds + 1
                if self._last_progress == self._progress
                else 0
            )
            if (
                self.state.stalled_rounds
                >= self.state.limits.no_progress_rounds
            ):
                self.begin_finalization(StopReason.NO_PROGRESS)
        self._last_progress = self._progress
        self.state.last_progress = self._last_progress

    def repair_diagnosis(
        self,
        *,
        validation: dict | None = None,
        candidate_preview: str | None = None,
        candidate_truncated: bool = False,
    ) -> None:
        """Persist rejection details before reserving a bounded repair.

        Failed model nodes may never checkpoint their messages. The event
        therefore retains a bounded candidate even when repairs are exhausted.
        """
        self.emit(
            "diagnosis.rejected",
            parent_span_id=self.state.id,
            data={
                "reason": StopReason.INVALID_DIAGNOSIS.value,
                **{
                    f"validation_{key}": value
                    for key, value in (validation or {}).items()
                },
                "candidate_preview": candidate_preview,
                "candidate_truncated": candidate_truncated,
            },
        )
        workflow = self.state.collaboration
        key = self._task_id.get()
        repairs = (
            workflow.repairs.get(key, 0)
            if workflow and key
            else self.state.diagnosis_repairs
        )
        if repairs >= self.state.limits.diagnosis_repairs:
            self.stop(StopReason.INVALID_DIAGNOSIS)
        if workflow is not None and key:
            workflow.repairs[key] = repairs + 1
        self.state.diagnosis_repairs += 1
        self.emit(
            "diagnosis.repair_requested",
            parent_span_id=self.state.id,
            data={"repair": self.state.diagnosis_repairs},
        )

    async def _model_attempt(
        self,
        call: Callable[[], Awaitable[T]],
        *,
        model_name: str,
        purpose: str,
        reserved_calls: int = 0,
    ) -> T:
        """Reserve one model call, enforce its timeout, and record
        usage/events.

        Failed admitted calls consume budget. Missing provider usage is
        recorded as unknown. Expected provider errors and timeouts stop the
        attempt; cancellation and unexpected exceptions propagate.
        """
        self.check_active()
        if self.state.model_calls >= self.state.limits.model_calls:
            self.stop(StopReason.MODEL_BUDGET)
        if (
            self.state.model_calls + reserved_calls
            >= self.state.limits.model_calls
        ):
            self.emit(
                "model.skipped",
                data={
                    "purpose": purpose,
                    "reason": "model_budget_reserved",
                    "reserved_calls": reserved_calls,
                },
            )
            raise ModelCallFailed(
                Failure("model_budget_reserved"), timed_out=False
            )
        self.state.model_calls += 1
        call_number = self.state.model_calls
        self.state.active_model = model_name
        span = uuid4()
        parent = self._tool_span.get() or self.state.id
        if purpose != "verification":
            self._model_spans[self._task_id.get()] = span
        started = monotonic()
        self.emit(
            "model.started",
            span_id=span,
            parent_span_id=parent,
            data={
                "model": model_name,
                "call": call_number,
                "purpose": purpose,
            },
        )
        reported = False
        cache_reported = False
        reasoning_reported = False
        call_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        try:
            async with asyncio.timeout(
                self.state.limits.model_timeout_seconds
            ):
                await self.checkpoint()
                response = await call()
            # Main-loop responses wrap AIMessage; auxiliary calls return it.
            messages = getattr(response, "result", (response,))
            cache_usage = []
            reasoning_usage = []
            for message in messages:
                if hasattr(message, "usage_metadata"):
                    cache_usage.append(_cache_usage(message))
                    usage = getattr(message, "usage_metadata", None) or {}
                    reasoning = (usage.get("output_token_details") or {}).get(
                        "reasoning"
                    )
                    reasoning_usage.append(
                        reasoning
                        if type(reasoning) is int
                        and 0 <= reasoning <= usage.get("output_tokens", 0)
                        else None
                    )
                item = getattr(message, "usage_metadata", None)
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
            cache_reported = bool(cache_usage) and all(
                item is not None for item in cache_usage
            )
            cache_hit = cache_miss = None
            reasoning_reported = bool(reasoning_usage) and all(
                item is not None for item in reasoning_usage
            )
            reasoning_tokens = (
                sum(reasoning_usage) if reasoning_reported else None
            )
            if reasoning_reported:
                self.state.usage.reported_reasoning_tokens += reasoning_tokens
            if cache_reported:
                cache_hit = sum(item[0] for item in cache_usage)
                cache_miss = sum(item[1] for item in cache_usage)
                self.state.usage.reported_cache_hit_tokens += cache_hit
                self.state.usage.reported_cache_miss_tokens += cache_miss
            self.emit(
                "model.completed",
                span_id=span,
                parent_span_id=parent,
                duration_ms=(monotonic() - started) * 1000,
                data={
                    "model": model_name,
                    "purpose": purpose,
                    "call": call_number,
                    "usage_reported": reported,
                    "cache_usage_reported": cache_reported,
                    "cache_hit_tokens": cache_hit,
                    "cache_miss_tokens": cache_miss,
                    "reasoning_usage_reported": reasoning_reported,
                    "reasoning_tokens": reasoning_tokens,
                    **{
                        key: value if reported else None
                        for key, value in call_usage.items()
                    },
                },
            )
            self.check_active()
            return response
        except TimeoutError:
            self.emit(
                "model.failed",
                span_id=span,
                parent_span_id=parent,
                duration_ms=(monotonic() - started) * 1000,
                data={"error_type": "TimeoutError"},
            )
            raise
        except asyncio.CancelledError:
            self.emit(
                "model.cancelled",
                span_id=span,
                parent_span_id=parent,
                duration_ms=(monotonic() - started) * 1000,
            )
            raise
        except Exception as exc:
            self.emit(
                "model.failed",
                span_id=span,
                parent_span_id=parent,
                duration_ms=(monotonic() - started) * 1000,
                data={
                    "error_type": type(exc).__name__,
                    "category": classify_failure(exc).category,
                    "status_code": classify_failure(exc).status_code,
                },
            )
            raise
        finally:
            if reasoning_reported:
                self.state.usage.reported_reasoning_model_calls += 1
            else:
                self.state.usage.unreported_reasoning_model_calls += 1
            if cache_reported:
                self.state.usage.reported_cache_model_calls += 1
            else:
                self.state.usage.unreported_cache_model_calls += 1
            if reported:
                self.state.usage.reported_model_calls += 1
            else:
                self.state.usage.unreported_model_calls += 1
            await self.checkpoint(check_cancel=False)

    def current_investigation_task(self):
        workflow = self.state.collaboration
        return workflow.tasks.get(self._task_id.get()) if workflow else None

    def check_task_tool_budget(self):
        record = self.current_investigation_task()
        if record is not None and record.tool_calls >= record.tool_limit:
            raise TaskToolBudgetExhausted(record.id)
        return record

    def task_tools_remaining(self):
        record = self.current_investigation_task()
        if record is None:
            return None
        return max(0, record.tool_limit - record.tool_calls)

    async def tool_call(
        self,
        name: str,
        arguments: dict[str, object],
        call: Callable[[], Awaitable[list[dict[str, object]]]],
        *,
        evidence_result: bool = True,
        remote: bool = True,
        retry_safe: bool = True,
    ) -> list[dict[str, object]]:
        """Enforce admission and retry only operations declared safe."""
        self.authorize_tool(name)
        self.check_task_tool_budget()
        signature = hashlib.sha256(
            json.dumps(
                [name, arguments], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if (
            self._signatures.get(signature, 0)
            >= self.state.limits.identical_tool_calls
        ):
            self.begin_finalization(StopReason.REPEATED_TOOL_CALL)
            raise InvestigationClosed()
        self._signatures[signature] = self._signatures.get(signature, 0) + 1
        self.state.tool_signatures = dict(self._signatures)
        task = asyncio.current_task()
        assert task is not None
        self._active.add(task)
        try:
            for attempt in range(self.state.limits.retry_attempts + 1):
                try:
                    result = await self._tool_attempt(
                        name,
                        signature,
                        call,
                        arguments,
                        evidence_result=evidence_result,
                        remote=remote,
                    )
                    for item in result:
                        if (
                            "id" in item
                            or "knowledge_ref" in item
                            or (
                                not evidence_result
                                and name
                                not in {
                                    "search_knowledge",
                                    "read_knowledge",
                                }
                            )
                        ):
                            fingerprint = hashlib.sha256(
                                json.dumps(
                                    item, sort_keys=True, default=str
                                ).encode()
                            ).hexdigest()
                            if fingerprint not in self._result_fingerprints:
                                self._result_fingerprints.add(fingerprint)
                                self._progress += 1
                    self.state.result_fingerprints = sorted(
                        self._result_fingerprints
                    )
                    self.state.progress_count = self._progress
                    return result
                except (TimeoutError, TelemetryClientError) as exc:
                    failure = classify_failure(exc)
                    if (
                        failure.retryable
                        and retry_safe
                        and attempt < self.state.limits.retry_attempts
                    ):
                        self.check_task_tool_budget()
                        await self._wait_retry(name, attempt, failure)
                        continue
                    if (
                        failure.retryable
                        or failure.category in {"invalid_request", "not_found"}
                    ):
                        raise ToolUnavailable(failure) from exc
                    self.stop(StopReason.TOOL_ERROR)
            raise AssertionError("retry loop must return or raise")
        finally:
            try:
                await self.checkpoint(check_cancel=False)
            finally:
                self._active.discard(task)

    async def _tool_attempt(
        self,
        name: str,
        signature: str,
        call: Callable[[], Awaitable[list[dict[str, object]]]],
        arguments: dict[str, object],
        *,
        evidence_result: bool,
        remote: bool,
    ) -> list[dict[str, object]]:
        """Reserve a shared attempt slot and record one backend execution."""
        self.authorize_tool(name)
        if sum(self.state.tool_calls.values()) >= self.state.limits.tool_calls:
            self.begin_finalization(StopReason.TOOL_BUDGET)
            raise InvestigationClosed()
        record = self.check_task_tool_budget()
        # Both reservations precede the checkpoint and any backend execution.
        if record is not None:
            record.tool_calls += 1
        self.state.tool_calls[name] = self.state.tool_calls.get(name, 0) + 1
        if remote:
            self.state.backend_attempts += 1
        span = uuid4()
        parent = self._model_spans.get(self._task_id.get(), self.state.id)
        started = monotonic()
        display = {}
        service = arguments.get("service")
        if isinstance(service, str):
            display["service"] = "".join(
                c for c in service if c.isprintable()
            )[:120]
        if remote:
            display.update(
                {
                    "investigation_start": (
                        self.state.investigation_start.isoformat()
                    ),
                    "investigation_end": (
                        self.state.investigation_end.isoformat()
                    ),
                }
            )
        self.emit(
            "tool.started",
            span_id=span,
            parent_span_id=parent,
            data={"tool": name, "arguments_sha256": signature, **display},
        )
        token = self._tool_span.set(span)
        try:
            async with asyncio.timeout(self.state.limits.tool_timeout_seconds):
                await self.checkpoint()
                result = await call()
            self.check_active()
            retained_ids = {str(key) for key in self.state.evidence_ids}
            self.emit(
                "tool.completed",
                span_id=span,
                parent_span_id=parent,
                duration_ms=(monotonic() - started) * 1000,
                data={
                    "tool": name,
                    "result_count": len(result),
                    "evidence_ids": [
                        str(item["id"])
                        for item in result
                        if evidence_result
                        and "id" in item
                        and str(item["id"]) in retained_ids
                    ],
                },
            )
            return result
        except TimeoutError:
            self.emit(
                "tool.failed",
                span_id=span,
                parent_span_id=parent,
                duration_ms=(monotonic() - started) * 1000,
                data={"tool": name, "error_type": "TimeoutError"},
            )
            raise
        except asyncio.CancelledError:
            self.emit(
                "tool.cancelled",
                span_id=span,
                parent_span_id=parent,
                duration_ms=(monotonic() - started) * 1000,
                data={"tool": name},
            )
            raise
        except Exception as exc:
            self.emit(
                "tool.failed",
                span_id=span,
                parent_span_id=parent,
                duration_ms=(monotonic() - started) * 1000,
                data={
                    "tool": name,
                    "error_type": type(exc).__name__,
                    "category": classify_failure(exc).category,
                    "status_code": classify_failure(exc).status_code,
                },
            )
            raise
        finally:
            self._tool_span.reset(token)

    async def drain(self) -> None:
        """Cancel sibling tool tasks before producing a terminal snapshot."""
        tasks = [
            task for task in self._active if task is not asyncio.current_task()
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
        self.state.elapsed_ms = (
            datetime.now(UTC) - self.state.started_at
        ).total_seconds() * 1000
        try:
            self.emit(
                "run.finished",
                span_id=self.state.id,
                duration_ms=self.state.elapsed_ms,
                data={
                    "status": status.value,
                    "reason": reason.value,
                    "model_calls": self.state.model_calls,
                    "tool_calls": sum(self.state.tool_calls.values()),
                    **self.state.usage.model_dump(),
                },
            )
        finally:
            self._close_trace()
