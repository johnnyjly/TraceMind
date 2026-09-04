"""Diagnosis orchestration, structured proposals, and completion checks."""

import asyncio
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from langgraph.errors import GraphRecursionError

from tracemind.agent.errors import (
    DiagnosisCancelled,
    DiagnosisError,
    DiagnosisValidationError,
)
from tracemind.agent.runtime.harness import RunHarness, RunStopped
from tracemind.agent.runtime.knowledge import (
    KnowledgeSearch, create_knowledge_tools,
)
from tracemind.agent.storage.io import run_io
from tracemind.agent.memory.lookup import (
    FeedbackWriter,
    GuidanceLookup,
    MemoryLookup,
)
from tracemind.agent.workflow.preparation import prepare_context
from tracemind.agent.runtime.runner import (
    AgentRunner,
    LangChainAgentRunner as LangChainAgentRunner,
    SYSTEM_PROMPT as SYSTEM_PROMPT,
)
from tracemind.agent.runtime.tools import EvidenceCollector, create_mcp_tools
from tracemind.agent.workflow.validation import validate_draft
from tracemind.mcp.client import ToolClient
from tracemind.models import Diagnosis, DiagnosisStatus, Evidence, Incident
from tracemind.models.memory import MemoryPolicy
from tracemind.models.knowledge import KnowledgePolicy
from tracemind.rag.model_bridge import Completion
from tracemind.models.result import DiagnosisDraft, DiagnosisRun
from tracemind.models.run import (
    RunLimits,
    RunState,
    RunStatus,
    StopReason,
    TOOL_NAMES,
    ToolName,
)


__all__ = [
    "DiagnosisService",
    "LangChainAgentRunner",
    "SYSTEM_PROMPT",
]


class DiagnosisService:
    """Own investigation setup, completion validation, and terminal cleanup."""

    def __init__(
        self,
        runner: AgentRunner,
        *,
        recursion_limit: int = 12,
        limits: RunLimits | None = None,
        allowed_tools: tuple[ToolName, ...] = TOOL_NAMES,
        event_directory: Path | None = None,
        memory_policy: MemoryPolicy | None = None,
        memory_lookup: MemoryLookup | None = None,
        guidance_lookup: GuidanceLookup | None = None,
        knowledge_policy: KnowledgePolicy | None = None,
        knowledge_search: KnowledgeSearch | None = None,
        knowledge_completion: Completion | None = None,
        knowledge_model_name: str = "rag-auxiliary",
        tool_client: ToolClient,
    ) -> None:
        """Configure telemetry dependencies, runner, and per-run controls."""
        self._runner = runner
        self._recursion_limit = recursion_limit
        self._limits = limits or RunLimits()
        self._allowed_tools = allowed_tools
        self._event_directory = event_directory
        self._memory_policy = memory_policy
        self._memory_lookup = memory_lookup
        self._guidance_lookup = guidance_lookup
        self._tool_client = tool_client
        self._knowledge_policy = knowledge_policy
        self._knowledge_search = knowledge_search
        self._knowledge_completion = knowledge_completion
        self._knowledge_model_name = knowledge_model_name

    async def diagnose(
        self,
        incident: Incident,
        *,
        start: datetime,
        end: datetime,
    ) -> Diagnosis:
        """Run an investigation and return only its accepted or partial
        diagnosis.

        Use diagnose_with_evidence when evidence and execution state are
        needed. Investigation failures propagate as DiagnosisError.
        """
        run = await self.diagnose_with_evidence(incident, start=start, end=end)
        return run.diagnosis

    async def diagnose_with_evidence(
        self,
        incident: Incident,
        *,
        start: datetime,
        end: datetime,
    ) -> DiagnosisRun:
        """Investigate a fixed, timezone-aware window with isolated run state.

        Return evidence and a diagnosis, including an inconclusive result
        for budget, repetition, recursion, or context exhaustion. Operational
        and contract failures raise DiagnosisError with partial evidence and
        state. Cancellation propagates. Once execution starts, terminal
        cleanup drains tools and finalizes the run.
        """
        if start.utcoffset() is None or end.utcoffset() is None:
            raise ValueError(
                "investigation timestamps must include a timezone"
            )
        if end <= start:
            raise ValueError("investigation end must be after its start")

        state = RunState(
            incident=incident,
            investigation_start=start,
            investigation_end=end,
            limits=self._limits.model_copy(deep=True),
            allowed_tools=self._allowed_tools,
            knowledge_policy=(self._knowledge_policy.model_copy(deep=True)
                              if self._knowledge_policy else None),
            memory_policy=self._memory_policy,
            learning_policy=(self._memory_policy
                             if self._guidance_lookup is not None else None),
        )
        collector = EvidenceCollector(max_bytes=state.limits.observation_bytes)
        return await self._execute(state, collector)

    async def run_persisted(
        self, state: RunState, evidence: list[Evidence], *,
        persist: Callable[[RunState, list[Evidence]], None],
        is_cancelled: Callable[[], bool],
        resume: bool,
        graph_resume: bool,
        feedback_writer: FeedbackWriter | None = None,
    ) -> DiagnosisRun:
        """Continue a leased run with its evidence and admission ledger."""
        collector: EvidenceCollector
        collector = EvidenceCollector(
            max_bytes=state.limits.observation_bytes,
            initial=evidence,
        )
        return await self._execute(
            state, collector, persist=persist, is_cancelled=is_cancelled,
            resume=resume, graph_resume=graph_resume,
            feedback_writer=feedback_writer,
        )

    async def _execute(
        self, state: RunState, collector: EvidenceCollector, *,
        persist: Callable[[RunState, list[Evidence]], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
        resume: bool = False,
        graph_resume: bool = False,
        feedback_writer: FeedbackWriter | None = None,
    ) -> DiagnosisRun:
        """Bind run-local tools and finalize either a fresh or resumed run."""
        incident = state.incident
        start = state.investigation_start
        end = state.investigation_end
        state.telemetry_transport = self._tool_client.transport

        async def persist_run(current: RunState) -> None:
            items = [item.model_copy(deep=True) for item in collector.evidence]
            await run_io(persist, current, items)

        harness = RunHarness(
            state, event_directory=self._event_directory,
            persist=persist_run if persist else None,
            is_cancelled=is_cancelled, resume=resume,
            graph_resume=graph_resume,
        )
        harness.validate_draft = lambda draft: validate_draft(
            draft, collector.evidence, start=start, end=end,
            knowledge=state.knowledge,
        )
        terminal_status = RunStatus.FAILED
        reason = StopReason.INTERNAL_ERROR
        try:
            async with asyncio.timeout(harness.remaining_seconds):
                tools = await create_mcp_tools(
                    self._tool_client, start=start, end=end,
                    collector=collector, harness=harness,
                    local_tools=create_knowledge_tools(
                        harness, search=self._knowledge_search,
                        completion=self._knowledge_completion,
                        model_name=self._knowledge_model_name,
                    ),
                )
                prompt = await prepare_context(
                    state, collector.evidence, harness,
                    memory_lookup=self._memory_lookup,
                    guidance_lookup=self._guidance_lookup,
                    learning_enabled=feedback_writer is not None,
                )
                await harness.checkpoint()
                draft = await self._runner.run(
                    prompt, tools,
                    recursion_limit=self._recursion_limit, harness=harness,
                )
                await harness.checkpoint()
                validate_draft(
                    draft, collector.evidence, start=start, end=end,
                    knowledge=state.knowledge,
                )
                state.feedback_updates = draft.feedback_updates
                harness.emit(
                    "diagnosis.accepted", parent_span_id=state.id,
                    data={
                        "status": draft.status.value,
                        "evidence_ids": [
                            str(item) for item in draft.evidence_ids
                        ],
                        "knowledge_refs": draft.knowledge_refs,
                    },
                )
                if (state.learning_policy is not None and state.user_message
                        and feedback_writer is not None):
                    outcome = await run_io(
                        feedback_writer, state.model_copy(deep=True))
                    state.feedback_updates = outcome.updates
                    if outcome.updates and not any(
                        e.kind == "memory.feedback_saved" for e in state.events
                    ):
                        harness.emit("memory.feedback_saved", data={
                            "applied": outcome.applied,
                            "skipped": outcome.skipped,
                            "rejected": outcome.rejected,
                            "rejected_count": len(outcome.rejected),
                            "guidance_ids": [str(i) for i in
                                             outcome.guidance_ids],
                            "memory_ids": [str(i) for i in outcome.memory_ids],
                        })
                    harness.check_active()
            if draft.status is DiagnosisStatus.COMPLETE:
                terminal_status = RunStatus.SUCCEEDED
                reason = StopReason.COMPLETED
            else:
                terminal_status = RunStatus.INCONCLUSIVE
                reason = StopReason.INSUFFICIENT_EVIDENCE
        except (RunStopped, TimeoutError, GraphRecursionError) as exc:
            if isinstance(exc, RunStopped):
                reason = exc.reason
            elif isinstance(exc, TimeoutError):
                reason = StopReason.DEADLINE
            else:
                reason = StopReason.RECURSION_LIMIT
            if reason is StopReason.CANCELLED:
                terminal_status = RunStatus.CANCELLED
            if reason in {
                StopReason.MODEL_BUDGET, StopReason.TOOL_BUDGET,
                StopReason.REPEATED_TOOL_CALL, StopReason.RECURSION_LIMIT,
                StopReason.CONTEXT_LIMIT,
                StopReason.NO_PROGRESS, StopReason.INVALID_DIAGNOSIS,
            }:
                terminal_status = RunStatus.INCONCLUSIVE
                draft = DiagnosisDraft(
                    status=DiagnosisStatus.INCONCLUSIVE,
                    root_cause=(
                        f"Investigation stopped ({reason.value}); "
                        "no root cause confirmed."
                    ),
                    confidence=0,
                )
            else:
                raise DiagnosisError(
                    f"Diagnosis stopped: {reason.value}",
                    state=state, evidence=collector.evidence,
                ) from exc
        except DiagnosisError as exc:
            reason = StopReason.INVALID_DIAGNOSIS
            validation = (exc.details() if isinstance(
                exc, DiagnosisValidationError
            ) else {})
            harness.emit("diagnosis.rejected", parent_span_id=state.id,
                         data={"reason": reason.value,
                               **{f"validation_{key}": value for key, value
                                  in validation.items()}})
            exc.state = state
            exc.evidence = collector.evidence
            raise
        except asyncio.CancelledError as exc:
            terminal_status = RunStatus.CANCELLED
            reason = StopReason.CANCELLED
            raise DiagnosisCancelled(state, collector.evidence) from exc
        except Exception as exc:
            raise DiagnosisError(
                "Diagnosis failed unexpectedly",
                state=state, evidence=collector.evidence,
            ) from exc
        finally:
            await harness.drain()
            state.evidence_ids = [item.id for item in collector.evidence]
            if terminal_status in {RunStatus.SUCCEEDED,
                                   RunStatus.INCONCLUSIVE}:
                state.diagnosis = Diagnosis(
                    incident_id=incident.id,
                    **draft.model_dump(exclude={"feedback_updates"}),
                )
            harness.finish(terminal_status, reason)
            await harness.checkpoint(check_cancel=False)

        return DiagnosisRun(
            diagnosis=state.diagnosis,
            evidence=collector.evidence,
            tool_calls=dict(state.tool_calls),
            state=state,
        )
