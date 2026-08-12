"""FastAPI routes and request-scoped diagnosis dependencies."""

from collections.abc import AsyncIterator
from typing import Annotated, Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException, Response, status
from pydantic import BaseModel

from tracemind.agent import (
    DiagnosisError,
    DiagnosisService,
    LangChainAgentRunner,
    ModelConfigurationError,
    create_chat_model,
)
from tracemind.api.schemas import DiagnosisRequest
from tracemind.config import Settings, get_settings
from tracemind.models import Diagnosis
from tracemind.models.run import StopReason
from tracemind.telemetry import (
    JaegerClient,
    OpenSearchClient,
    PrometheusClient,
)


class HealthResponse(BaseModel):
    """Liveness response containing the literal status ok."""

    status: Literal["ok"]


async def get_request_settings() -> Settings:
    """Expose cached settings as an overridable FastAPI dependency."""
    return get_settings()


async def get_agent_runner(
    settings: Annotated[Settings, Depends(get_request_settings)],
) -> LangChainAgentRunner:
    """Build a model runner; report missing provider configuration as HTTP
    503.
    """
    try:
        return LangChainAgentRunner(create_chat_model(settings))
    except ModelConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


async def get_diagnosis_service(
    settings: Annotated[Settings, Depends(get_request_settings)],
    runner: Annotated[LangChainAgentRunner, Depends(get_agent_runner)],
) -> AsyncIterator[DiagnosisService]:
    """Yield a configured service and close its HTTP client after the
    request.
    """
    async with httpx.AsyncClient(timeout=20) as http_client:
        yield DiagnosisService(
            PrometheusClient(str(settings.prometheus_url), http_client),
            JaegerClient(str(settings.jaeger_url), http_client),
            OpenSearchClient(
                str(settings.opensearch_url),
                settings.opensearch_logs_index,
                http_client,
            ),
            runner,
            recursion_limit=settings.agent_recursion_limit,
            limits=settings.run_limits,
            allowed_tools=settings.agent_allowed_tools,
            event_directory=settings.agent_trace_directory,
        )


def create_app() -> FastAPI:
    """Build the application with health and synchronous diagnosis routes."""
    application = FastAPI(
        title="TraceMind",
        description="Evidence-driven incident diagnosis for microservices.",
        version="0.1.0",
    )

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        """Return process liveness without checking model or backend
        availability.
        """
        return HealthResponse(status="ok")

    @application.post(
        "/api/v1/diagnoses",
        response_model=Diagnosis,
        tags=["diagnosis"],
    )
    async def diagnose(
        request: DiagnosisRequest,
        response: Response,
        service: Annotated[DiagnosisService, Depends(get_diagnosis_service)],
    ) -> Diagnosis:
        """Investigate the request and return a diagnosis with run headers.

        Translate operational timeouts to HTTP 504 and other diagnosis
        failures to HTTP 502. Error detail remains a string; available
        run metadata is returned in headers.
        """
        try:
            run = await service.diagnose_with_evidence(
                request.incident,
                start=request.investigation_start,
                end=request.investigation_end,
            )
            if run.state is not None and run.state.stop_reason is not None:
                response.headers["X-TraceMind-Run-ID"] = str(run.state.id)
                response.headers["X-TraceMind-Stop-Reason"] = (
                    run.state.stop_reason.value
                )
            return run.diagnosis
        except DiagnosisError as exc:
            reason = exc.state.stop_reason if exc.state else None
            timed_out = reason in {
                StopReason.DEADLINE,
                StopReason.MODEL_TIMEOUT,
                StopReason.TOOL_TIMEOUT,
            }
            headers = {}
            if exc.state is not None:
                headers["X-TraceMind-Run-ID"] = str(exc.state.id)
            if reason is not None:
                headers["X-TraceMind-Stop-Reason"] = reason.value
            raise HTTPException(
                status_code=(
                    status.HTTP_504_GATEWAY_TIMEOUT if timed_out
                    else status.HTTP_502_BAD_GATEWAY
                ),
                detail=str(exc),
                headers=headers,
            ) from exc

    return application


app = create_app()
