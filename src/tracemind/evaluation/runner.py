"""Live diagnosis execution followed by external ground-truth scoring."""

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
from tracemind.evaluation.evaluator import evaluate_diagnosis
from tracemind.evaluation.models import EvaluationCase, EvaluationReport
from tracemind.mcp.client import open_telemetry_backend
from tracemind.models.memory import MemorySnapshot


async def run_service_evaluation(
    case: EvaluationCase, service: DiagnosisService, *,
    start: datetime, end: datetime, model_provider: str, model_name: str,
    execution_mode: str = "live",
    memory_corpus_id: str | None = None,
) -> EvaluationReport:
    """Retain operational failures and cancellations in a scored report.

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
            case.incident, start=start, end=end,
        )
    except DiagnosisError as exc:
        return EvaluationReport(
            case_id=case.id, model_provider=model_provider,
            model_name=model_name, investigation_start=start,
            investigation_end=end, evidence=exc.evidence,
            run_state=exc.state, error_type=type(exc).__name__,
            execution_mode=execution_mode,
            elapsed_ms=(perf_counter() - started) * 1000,
            telemetry_transport=(exc.state.telemetry_transport
                                 if exc.state else "direct"),
            **memory_metadata,
        )
    except asyncio.CancelledError as exc:
        return EvaluationReport(
            case_id=case.id, model_provider=model_provider,
            model_name=model_name, investigation_start=start,
            investigation_end=end,
            evidence=exc.evidence if isinstance(exc, DiagnosisCancelled)
            else [],
            run_state=exc.state if isinstance(exc, DiagnosisCancelled)
            else None,
            error_type="CancelledError", execution_mode=execution_mode,
            telemetry_transport=(exc.state.telemetry_transport
                                 if isinstance(exc, DiagnosisCancelled)
                                 and exc.state else "direct"),
            elapsed_ms=(perf_counter() - started) * 1000,
            **memory_metadata,
        )
    elapsed_ms = (perf_counter() - started) * 1000
    return EvaluationReport(
        case_id=case.id, model_provider=model_provider, model_name=model_name,
        investigation_start=start, investigation_end=end,
        diagnosis=run.diagnosis, evidence=run.evidence,
        result=evaluate_diagnosis(case, run, elapsed_ms=elapsed_ms),
        run_state=run.state, execution_mode=execution_mode,
        elapsed_ms=elapsed_ms,
        telemetry_transport=(run.state.telemetry_transport
                             if run.state else "direct"),
        **memory_metadata,
    )


async def run_live_evaluation(
    case: EvaluationCase,
    settings: Settings,
    *,
    start: datetime,
    end: datetime,
    callbacks: Sequence[BaseCallbackHandler] | None = None,
    memory_snapshot: MemorySnapshot | None = None,
) -> EvaluationReport:
    """Run a live investigation and score its result against the case.

    Constrain tool calls by both application and case limits. Ground
    truth is used only after diagnosis. Operational failures retain their
    partial state in a report; no report file is written here.
    """
    policy, lookup = evaluation_memory(
        memory_snapshot, settings.memory_policy, start=start,
        max_records=settings.agent_memory_max_records,
        context_bytes=settings.agent_memory_context_bytes,
    )
    corpus_id = memory_snapshot.fingerprint if memory_snapshot else None
    try:
        runner = build_runner(settings, callbacks=callbacks)
    except ModelConfigurationError as exc:
        return EvaluationReport(
            case_id=case.id, model_provider=settings.llm_provider,
            model_name=settings.llm_model,
            investigation_start=start, investigation_end=end,
            evidence=[], error_type=type(exc).__name__,
            memory_mode="frozen" if corpus_id else "off",
            memory_corpus_id=corpus_id,
            telemetry_transport=settings.telemetry_transport,
        )
    limits = settings.run_limits
    limits.tool_calls = min(limits.tool_calls, case.max_tool_calls)
    async with open_telemetry_backend(settings) as tool_client:
        service = build_service(
            settings, runner, tool_client, limits=limits,
            memory_policy=policy, memory_lookup=lookup,
        )
        return await run_service_evaluation(
            case, service, start=start, end=end,
            model_provider=settings.llm_provider,
            model_name=settings.llm_model,
            memory_corpus_id=corpus_id,
        )
