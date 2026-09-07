"""Diagnosis task execution; all scoring belongs to Langfuse evaluators."""

import asyncio
from collections.abc import Sequence
from datetime import datetime
from time import perf_counter

from langchain_core.callbacks import BaseCallbackHandler

from tracemind.agent.errors import (
    DiagnosisCancelled,
    DiagnosisError,
    ModelConfigurationError,
)
from tracemind.agent.factory import build_runner, build_service
from tracemind.agent.memory.lookup import evaluation_memory
from tracemind.agent.service import DiagnosisService
from tracemind.config import Settings
from tracemind.evaluation.models import ExperimentOutput
from tracemind.mcp.client import open_telemetry_backend
from tracemind.models.memory import MemorySnapshot


async def execute_service(
    incident,
    service: DiagnosisService,
    *,
    start: datetime,
    end: datetime,
    model_provider: str,
    model_name: str,
    execution_mode: str = "live",
    memory_corpus_id: str | None = None,
) -> ExperimentOutput:
    """Retain operational failures and cancellations as observed task output.

    Ground truth is accessed after execution and never enters the service.
    Service-reported failures retain partial state; errors outside that
    boundary propagate rather than becoming scores.
    """
    started = perf_counter()
    memory_metadata = {
        "memory_mode": "frozen" if memory_corpus_id else "off",
        "memory_corpus_id": memory_corpus_id,
    }
    try:
        run = await service.diagnose_with_evidence(
            incident,
            start=start,
            end=end,
        )
    except DiagnosisError as exc:
        return ExperimentOutput(
            model_provider=model_provider,
            model_name=model_name,
            investigation_start=start,
            investigation_end=end,
            evidence=exc.evidence,
            run_state=exc.state,
            error_type=type(exc).__name__,
            execution_mode=execution_mode,
            elapsed_ms=(perf_counter() - started) * 1000,
            telemetry_transport=(
                exc.state.telemetry_transport if exc.state else "direct"
            ),
            **memory_metadata,
        )
    except asyncio.CancelledError as exc:
        return ExperimentOutput(
            model_provider=model_provider,
            model_name=model_name,
            investigation_start=start,
            investigation_end=end,
            evidence=(
                exc.evidence if isinstance(exc, DiagnosisCancelled) else []
            ),
            run_state=(
                exc.state if isinstance(exc, DiagnosisCancelled) else None
            ),
            error_type="CancelledError",
            execution_mode=execution_mode,
            telemetry_transport=(
                exc.state.telemetry_transport
                if isinstance(exc, DiagnosisCancelled) and exc.state
                else "direct"
            ),
            elapsed_ms=(perf_counter() - started) * 1000,
            **memory_metadata,
        )
    elapsed_ms = (perf_counter() - started) * 1000
    return ExperimentOutput(
        model_provider=model_provider,
        model_name=model_name,
        investigation_start=start,
        investigation_end=end,
        diagnosis=run.diagnosis,
        knowledge=([r.model_dump(mode="json") for r in run.state.knowledge]
                   if run.state else []),
        evidence=run.evidence,
        run_state=run.state,
        execution_mode=execution_mode,
        elapsed_ms=elapsed_ms,
        telemetry_transport=(
            run.state.telemetry_transport if run.state else "direct"
        ),
        **memory_metadata,
    )


async def execute_agent(
    incident,
    settings: Settings,
    *,
    start: datetime,
    end: datetime,
    callbacks: Sequence[BaseCallbackHandler] | None = None,
    max_tool_calls: int | None = None,
    memory_snapshot: MemorySnapshot | None = None,
) -> ExperimentOutput:
    """Run a live investigation without access to evaluation labels.

    Constrain tool calls by both application and case limits. Ground
    truth is used only after diagnosis. Operational failures retain their
    partial state in the experiment output.
    """
    policy, lookup = evaluation_memory(
        memory_snapshot,
        settings.memory_policy,
        start=start,
        max_records=settings.agent_memory_max_records,
        context_bytes=settings.agent_memory_context_bytes,
    )
    corpus_id = memory_snapshot.fingerprint if memory_snapshot else None
    try:
        runner = build_runner(settings, callbacks=callbacks)
    except ModelConfigurationError as exc:
        return ExperimentOutput(
            model_provider=settings.llm_provider,
            model_name=settings.llm_model,
            investigation_start=start,
            investigation_end=end,
            evidence=[],
            error_type=type(exc).__name__,
            memory_mode="frozen" if corpus_id else "off",
            memory_corpus_id=corpus_id,
            telemetry_transport=settings.telemetry_transport,
        )
    limits = settings.run_limits
    if max_tool_calls is not None:
        limits.tool_calls = min(limits.tool_calls, max_tool_calls)
    async with open_telemetry_backend(settings) as tool_client:
        service = build_service(
            settings,
            runner,
            tool_client,
            limits=limits,
            memory_policy=policy,
            memory_lookup=lookup,
            callbacks=callbacks,
        )
        return await execute_service(
            incident,
            service,
            start=start,
            end=end,
            model_provider=settings.llm_provider,
            model_name=settings.llm_model,
            memory_corpus_id=corpus_id,
        )
