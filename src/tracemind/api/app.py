from collections.abc import AsyncIterator
from typing import Annotated, Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException, status
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
from tracemind.telemetry import JaegerClient, OpenSearchClient, PrometheusClient


class HealthResponse(BaseModel):
    status: Literal["ok"]


async def get_request_settings() -> Settings:
    return get_settings()


async def get_agent_runner(
    settings: Annotated[Settings, Depends(get_request_settings)],
) -> LangChainAgentRunner:
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
        )


def create_app() -> FastAPI:
    application = FastAPI(
        title="TraceMind",
        description="Evidence-driven incident diagnosis for microservices.",
        version="0.1.0",
    )

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.post(
        "/api/v1/diagnoses",
        response_model=Diagnosis,
        tags=["diagnosis"],
    )
    async def diagnose(
        request: DiagnosisRequest,
        service: Annotated[DiagnosisService, Depends(get_diagnosis_service)],
    ) -> Diagnosis:
        try:
            return await service.diagnose(
                request.incident,
                start=request.investigation_start,
                end=request.investigation_end,
            )
        except DiagnosisError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc

    return application


app = create_app()
