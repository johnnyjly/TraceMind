"""FastAPI routes and request-scoped diagnosis dependencies."""

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from typing import Annotated, Literal
from uuid import UUID, uuid4

import httpx
from fastapi import (
    Depends, FastAPI, Header, HTTPException, Query, Request, Response, status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from tracemind.agent import (
    DiagnosisError,
    DiagnosisService,
    LangChainAgentRunner,
    ModelConfigurationError,
    create_chat_model,
)
from tracemind.agent.persistence import (
    RunConflict, RunNotFound, RunRepository, StoredRun,
)
from tracemind.api.schemas import (
    DiagnosisRequest, FollowUpRequest, InvestigationCreated,
    InvestigationResponse, InvestigationTurn, RunStatusResponse,
)
from tracemind.api.progress import stream_progress
from tracemind.agent.llm import create_fallback_model
from tracemind.config import Settings, get_settings
from tracemind.models import Diagnosis
from tracemind.models.run import RunPurpose, RunState, RunStatus, StopReason
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
        return LangChainAgentRunner(
            create_chat_model(settings),
            fallback_model=create_fallback_model(settings),
        )
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


async def get_run_repository(
    settings: Annotated[Settings, Depends(get_request_settings)],
) -> RunRepository:
    """Open the configured SQLite run store for lifecycle requests."""
    return RunRepository(settings.agent_run_database)


def run_status(record: StoredRun) -> RunStatusResponse:
    """Project a durable snapshot without exposing raw observations."""
    state = record.state
    return RunStatusResponse(
        id=state.id, status=state.status, stop_reason=state.stop_reason,
        investigation_id=state.investigation_id,
        parent_run_id=state.parent_run_id, purpose=state.purpose,
        cancel_requested=record.cancel_requested, attempt=record.attempt,
        model_calls=state.model_calls,
        tool_calls=sum(state.tool_calls.values()),
        evidence_count=len(record.evidence),
        started_at=state.started_at if record.attempt else None,
        finished_at=state.finished_at,
    )


def request_fingerprint(
    request: BaseModel, operation: str | None = None,
) -> str:
    """Hash validated input, omitting generated incident IDs."""
    excluded = None
    if isinstance(request, DiagnosisRequest):
        if "id" not in request.incident.model_fields_set:
            excluded = {"incident": {"id"}}
    payload = request.model_dump(mode="json", exclude=excluded)
    if operation is not None:
        payload = {"operation": operation, "body": payload}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def create_app() -> FastAPI:
    """Build health, synchronous diagnosis, and persistent run routes."""
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

    @application.post(
        "/api/v1/runs", status_code=status.HTTP_202_ACCEPTED,
        response_model=RunStatusResponse, tags=["runs"],
    )
    async def create_run(
        request: DiagnosisRequest,
        response: Response,
        settings: Annotated[Settings, Depends(get_request_settings)],
        repository: Annotated[RunRepository, Depends(get_run_repository)],
        _runner: Annotated[LangChainAgentRunner, Depends(get_agent_runner)],
        idempotency_key: Annotated[
            str | None, Header(alias="Idempotency-Key", max_length=128)
        ] = None,
    ) -> RunStatusResponse:
        """Queue a diagnosis for the independent persistent worker."""
        state = RunState(
            incident=request.incident,
            investigation_start=request.investigation_start,
            investigation_end=request.investigation_end,
            limits=settings.run_limits,
            allowed_tools=settings.agent_allowed_tools,
            status=RunStatus.QUEUED,
        )
        request_hash = request_fingerprint(request)
        try:
            record = repository.create(
                state, request_hash=request_hash,
                idempotency_key=idempotency_key,
            )
        except RunConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        response.headers["Location"] = f"/api/v1/runs/{record.state.id}"
        return run_status(record)

    @application.get(
        "/api/v1/runs/{run_id}", response_model=RunStatusResponse,
        tags=["runs"],
    )
    async def get_run(
        run_id: UUID,
        repository: Annotated[RunRepository, Depends(get_run_repository)],
    ) -> RunStatusResponse:
        """Read persisted status, counters, and cancellation intent."""
        try:
            return run_status(repository.get(run_id))
        except RunNotFound as exc:
            raise HTTPException(
                status_code=404, detail="run not found"
            ) from exc

    @application.get("/api/v1/runs/{run_id}/result", tags=["runs"])
    async def get_run_result(
        run_id: UUID,
        repository: Annotated[RunRepository, Depends(get_run_repository)],
    ) -> dict:
        """Return the terminal diagnosis with its retained evidence."""
        try:
            record = repository.get(run_id)
        except RunNotFound as exc:
            raise HTTPException(
                status_code=404, detail="run not found"
            ) from exc
        if record.result is None:
            raise HTTPException(
                status_code=409, detail="run has no diagnosis result"
            )
        return record.result.model_dump(mode="json")

    @application.post(
        "/api/v1/runs/{run_id}/cancel", status_code=202,
        response_model=RunStatusResponse, tags=["runs"],
    )
    async def cancel_run(
        run_id: UUID,
        repository: Annotated[RunRepository, Depends(get_run_repository)],
    ) -> RunStatusResponse:
        """Persist a cancellation request for the current worker."""
        try:
            return run_status(repository.cancel(run_id))
        except RunNotFound as exc:
            raise HTTPException(
                status_code=404, detail="run not found"
            ) from exc

    @application.post(
        "/api/v1/runs/{run_id}/resume", status_code=202,
        response_model=RunStatusResponse, tags=["runs"],
    )
    async def resume_run(
        run_id: UUID,
        repository: Annotated[RunRepository, Depends(get_run_repository)],
    ) -> RunStatusResponse:
        """Requeue an interrupted run with an expired worker lease."""
        try:
            return run_status(repository.resume(run_id))
        except RunNotFound as exc:
            raise HTTPException(
                status_code=404, detail="run not found"
            ) from exc
        except RunConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post(
        "/api/v1/investigations", status_code=202,
        response_model=InvestigationCreated, tags=["investigations"],
    )
    async def create_investigation(
        request: DiagnosisRequest, response: Response,
        settings: Annotated[Settings, Depends(get_request_settings)],
        repository: Annotated[RunRepository, Depends(get_run_repository)],
        _runner: Annotated[LangChainAgentRunner, Depends(get_agent_runner)],
        idempotency_key: Annotated[
            str | None, Header(alias="Idempotency-Key", max_length=128)
        ] = None,
    ) -> InvestigationCreated:
        """Start autonomous diagnosis with optional later human follow-up."""
        state = RunState(
            investigation_id=uuid4(), incident=request.incident,
            investigation_start=request.investigation_start,
            investigation_end=request.investigation_end,
            limits=settings.run_limits,
            allowed_tools=settings.agent_allowed_tools,
            status=RunStatus.QUEUED,
        )
        try:
            record = repository.create(
                state, request_hash=request_fingerprint(request,
                                                        "investigation"),
                idempotency_key=idempotency_key, new_investigation=True,
            )
        except RunConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        response.headers["Location"] = (
            f"/api/v1/investigations/{record.state.investigation_id}"
        )
        return InvestigationCreated(id=record.state.investigation_id,
                                    run=run_status(record))

    @application.get(
        "/api/v1/investigations/{investigation_id}",
        response_model=InvestigationResponse, tags=["investigations"],
    )
    async def get_investigation(
        investigation_id: UUID,
        repository: Annotated[RunRepository, Depends(get_run_repository)],
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ) -> InvestigationResponse:
        """Read the preserved incident and chronological round history."""
        try:
            record = repository.get_investigation(
                investigation_id, offset=offset, limit=limit,
            )
        except RunNotFound as exc:
            raise HTTPException(
                status_code=404, detail="investigation not found"
            ) from exc
        return InvestigationResponse(
            id=record.id, incident=record.incident,
            created_at=record.created_at,
            next_offset=record.next_offset,
            turns=[InvestigationTurn(
                run_id=s.id, parent_run_id=s.parent_run_id,
                purpose=s.purpose, status=s.status, stop_reason=s.stop_reason,
                investigation_start=s.investigation_start,
                investigation_end=s.investigation_end,
                user_message=s.user_message, diagnosis=s.diagnosis,
            ) for s in record.runs],
        )

    @application.post(
        "/api/v1/investigations/{investigation_id}/messages", status_code=202,
        response_model=RunStatusResponse, tags=["investigations"],
    )
    async def follow_up(
        investigation_id: UUID, request: FollowUpRequest, response: Response,
        settings: Annotated[Settings, Depends(get_request_settings)],
        repository: Annotated[RunRepository, Depends(get_run_repository)],
        _runner: Annotated[LangChainAgentRunner, Depends(get_agent_runner)],
        idempotency_key: Annotated[
            str | None, Header(alias="Idempotency-Key", max_length=128)
        ] = None,
    ) -> RunStatusResponse:
        """Queue a separately budgeted explanation or hypothesis check."""
        try:
            record = repository.follow_up(
                investigation_id, message=request.message,
                purpose=RunPurpose(request.purpose),
                limits=settings.run_limits,
                allowed_tools=settings.agent_allowed_tools,
                start=request.investigation_start,
                end=request.investigation_end,
                request_hash=request_fingerprint(request,
                                                 str(investigation_id)),
                idempotency_key=idempotency_key,
            )
        except RunNotFound as exc:
            raise HTTPException(
                status_code=404, detail="investigation not found"
            ) from exc
        except RunConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        response.headers["Location"] = f"/api/v1/runs/{record.state.id}"
        return run_status(record)

    @application.get("/api/v1/runs/{run_id}/events", tags=["runs"])
    async def run_events(
        run_id: UUID, request: Request,
        repository: Annotated[RunRepository, Depends(get_run_repository)],
        after: Annotated[int, Query(ge=0)] = 0,
        last_event_id: Annotated[
            str | None, Header(alias="Last-Event-ID")
        ] = None,
    ) -> StreamingResponse:
        """Subscribe to readable progress or replay from a durable cursor."""
        if last_event_id is not None:
            if (not last_event_id.isascii() or not last_event_id.isdigit()
                    or len(last_event_id) > 20):
                raise HTTPException(status_code=400, detail="invalid event ID")
            after = int(last_event_id)
        try:
            snapshot = await asyncio.to_thread(
                repository.event_snapshot, run_id,
            )
        except RunNotFound as exc:
            raise HTTPException(
                status_code=404, detail="run not found"
            ) from exc
        latest = snapshot.events[-1].sequence if snapshot.events else 0
        if after > latest:
            raise HTTPException(status_code=409, detail="event ID is ahead")
        return StreamingResponse(
            stream_progress(repository, run_id, request, after=after),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return application


app = create_app()
