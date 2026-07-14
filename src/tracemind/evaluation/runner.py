from collections.abc import Sequence
from datetime import datetime
from time import perf_counter

import httpx
from langchain_core.callbacks import BaseCallbackHandler

from tracemind.agent import DiagnosisService, LangChainAgentRunner, create_chat_model
from tracemind.config import Settings
from tracemind.evaluation.evaluator import evaluate_diagnosis
from tracemind.evaluation.models import EvaluationCase, EvaluationReport
from tracemind.telemetry import JaegerClient, OpenSearchClient, PrometheusClient


async def run_live_evaluation(
    case: EvaluationCase,
    settings: Settings,
    *,
    start: datetime,
    end: datetime,
    callbacks: Sequence[BaseCallbackHandler] | None = None,
) -> EvaluationReport:
    model = create_chat_model(settings)
    async with httpx.AsyncClient(timeout=20) as http_client:
        service = DiagnosisService(
            PrometheusClient(str(settings.prometheus_url), http_client),
            JaegerClient(str(settings.jaeger_url), http_client),
            OpenSearchClient(
                str(settings.opensearch_url),
                settings.opensearch_logs_index,
                http_client,
            ),
            LangChainAgentRunner(model, callbacks=callbacks),
            recursion_limit=settings.agent_recursion_limit,
        )
        started = perf_counter()
        run = await service.diagnose_with_evidence(
            case.incident,
            start=start,
            end=end,
        )
        elapsed_ms = (perf_counter() - started) * 1_000

    result = evaluate_diagnosis(case, run, elapsed_ms=elapsed_ms)
    return EvaluationReport(
        case_id=case.id,
        model_provider=settings.llm_provider,
        model_name=settings.llm_model,
        investigation_start=start,
        investigation_end=end,
        diagnosis=run.diagnosis,
        evidence=run.evidence,
        result=result,
    )
