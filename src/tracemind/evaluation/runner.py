"""Live diagnosis execution followed by external ground-truth scoring."""

import asyncio
from collections.abc import Sequence
from datetime import datetime
from time import perf_counter

import httpx
from langchain_core.callbacks import BaseCallbackHandler

from tracemind.agent import (
    DiagnosisError,
    DiagnosisService,
    LangChainAgentRunner,
    create_chat_model,
)
from tracemind.agent.errors import DiagnosisCancelled, ModelConfigurationError
from tracemind.config import Settings
from tracemind.agent.llm import create_fallback_model
from tracemind.evaluation.evaluator import evaluate_diagnosis
from tracemind.evaluation.models import EvaluationCase, EvaluationReport
from tracemind.telemetry import (
    JaegerClient,
    OpenSearchClient,
    PrometheusClient,
)


async def run_service_evaluation(
    case: EvaluationCase, service: DiagnosisService, *,
    start: datetime, end: datetime, model_provider: str, model_name: str,
    execution_mode: str = "live",
) -> EvaluationReport:
    """Retain operational failures and cancellations in a scored report.

    Ground truth is accessed after execution and never enters the service.
    Service-reported failures retain partial state; errors outside that
    boundary propagate rather than becoming scores.
    """
    started = perf_counter()
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
            elapsed_ms=(perf_counter() - started) * 1000,
        )
    elapsed_ms = (perf_counter() - started) * 1000
    return EvaluationReport(
        case_id=case.id, model_provider=model_provider, model_name=model_name,
        investigation_start=start, investigation_end=end,
        diagnosis=run.diagnosis, evidence=run.evidence,
        result=evaluate_diagnosis(case, run, elapsed_ms=elapsed_ms),
        run_state=run.state, execution_mode=execution_mode,
        elapsed_ms=elapsed_ms,
    )


async def run_live_evaluation(
    case: EvaluationCase,
    settings: Settings,
    *,
    start: datetime,
    end: datetime,
    callbacks: Sequence[BaseCallbackHandler] | None = None,
) -> EvaluationReport:
    """Run a live investigation and score its result against the case.

    Constrain tool calls by both application and case limits. Ground
    truth is used only after diagnosis. Operational failures retain their
    partial state in a report; no report file is written here.
    """
    try:
        model = create_chat_model(settings)
        fallback = create_fallback_model(settings)
    except ModelConfigurationError as exc:
        return EvaluationReport(
            case_id=case.id, model_provider=settings.llm_provider,
            model_name=settings.llm_model,
            investigation_start=start, investigation_end=end,
            evidence=[], error_type=type(exc).__name__,
        )
    limits = settings.run_limits
    limits.tool_calls = min(limits.tool_calls, case.max_tool_calls)
    async with httpx.AsyncClient(timeout=20) as http_client:
        service = DiagnosisService(
            PrometheusClient(str(settings.prometheus_url), http_client),
            JaegerClient(str(settings.jaeger_url), http_client),
            OpenSearchClient(
                str(settings.opensearch_url),
                settings.opensearch_logs_index,
                http_client,
            ),
            LangChainAgentRunner(
                model, callbacks=callbacks,
                fallback_model=fallback,
            ),
            recursion_limit=settings.agent_recursion_limit,
            limits=limits,
            allowed_tools=settings.agent_allowed_tools,
            event_directory=settings.agent_trace_directory,
        )
        return await run_service_evaluation(
            case, service, start=start, end=end,
            model_provider=settings.llm_provider,
            model_name=settings.llm_model,
        )
