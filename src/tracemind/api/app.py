"""FastAPI routes and request-scoped diagnosis dependencies."""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
import hashlib
import json
from threading import Lock
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Header,
    Query,
    Request,
    Response,
    status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from tracemind.agent.errors import DiagnosisError, ModelConfigurationError
from tracemind.agent.factory import build_runner, build_service
from tracemind.agent.storage.memory_repository import (
    MemoryNotFound,
    MemoryRepository,
)
from tracemind.agent.storage.run_repository import RunRepository
from tracemind.agent.runtime.runner import LangChainAgentRunner
from tracemind.agent.service import DiagnosisService
from tracemind.api.progress import stream_progress
from tracemind.api.schemas import (
    DiagnosisRequest,
    FollowUpRequest,
    InvestigationCreated,
    InvestigationResponse,
    InvestigationTurn,
    RunStatusResponse,
)
from tracemind.config import Settings, get_settings
from tracemind.mcp.client import open_telemetry_backend
from tracemind.models import Diagnosis, Incident
from tracemind.models.memory import (
    GuidanceRecord,
    GuidanceRevision,
    MemoryCreate,
    MemoryPolicy,
    MemoryRecord,
    MemoryRevision,
    MemoryScope,
    MemorySnapshot,
    ScopeName,
)
from tracemind.models.run import RunPurpose, RunState, RunStatus, StopReason
from tracemind.models.storage import RunConflict, RunNotFound, StoredRun


class HealthResponse(BaseModel):
    """Liveness response containing the literal status ok."""

    status: Literal["ok"]


async def get_request_settings() -> Settings:
    """Expose cached settings as an overridable FastAPI dependency."""
    return get_settings()


def get_agent_runner(
    settings: Annotated[Settings, Depends(get_request_settings)],
) -> LangChainAgentRunner:
    """Build a model runner; report missing provider configuration as HTTP
    503.
    """
    try:
        return build_runner(settings)
    except ModelConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


async def get_diagnosis_service(
    request: Request,
    settings: Annotated[Settings, Depends(get_request_settings)],
    runner: Annotated[LangChainAgentRunner, Depends(get_agent_runner)],
) -> AsyncIterator[DiagnosisService]:
    """Yield a configured service and close its HTTP client after the
    request.
    """
    from tracemind.agent.storage.io import run_io

    async with open_telemetry_backend(settings) as tool_client:
        memory = (
            MemoryRepository(
                (await run_io(get_run_repository, settings, request)).database
            )
            if settings.agent_memory_enabled
            else None
        )
        yield build_service(
            settings,
            runner,
            tool_client,
            memory_policy=settings.memory_policy,
            memory_lookup=memory.snapshot if memory is not None else None,
            guidance_lookup=(
                memory.guidance_snapshot if memory is not None else None
            ),
        )


def get_run_repository(
    settings: Annotated[Settings, Depends(get_request_settings)],
    request: Request,
) -> RunRepository:
    """Initialize each app-owned store once, in the request thread pool."""
    with request.app.state.repository_lock:
        stores = request.app.state.repositories
        path = settings.agent_run_database
        if path not in stores:
            stores[path] = RunRepository(path)
        return stores[path]


def run_status(record: StoredRun) -> RunStatusResponse:
    """Project a durable snapshot without exposing raw observations."""
    state = record.state
    return RunStatusResponse(
        id=state.id,
        status=state.status,
        stop_reason=state.stop_reason,
        investigation_id=state.investigation_id,
        parent_run_id=state.parent_run_id,
        purpose=state.purpose,
        cancel_requested=record.cancel_requested,
        attempt=record.attempt,
        model_calls=state.model_calls,
        tool_calls=sum(state.tool_calls.values()),
        evidence_count=len(record.evidence),
        started_at=state.started_at if record.attempt else None,
        finished_at=state.finished_at,
    )


def request_fingerprint(
    request: BaseModel,
    operation: str | None = None,
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

    application.state.repositories = {}
    application.state.repository_lock = Lock()

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
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
                    status.HTTP_504_GATEWAY_TIMEOUT
                    if timed_out
                    else status.HTTP_502_BAD_GATEWAY
                ),
                detail=str(exc),
                headers=headers,
            ) from exc

    @application.post(
        "/api/v1/runs",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=RunStatusResponse,
        tags=["runs"],
    )
    def create_run(
        request: DiagnosisRequest,
        response: Response,
        settings: Annotated[Settings, Depends(get_request_settings)],
        repository: Annotated[RunRepository, Depends(get_run_repository)],
        idempotency_key: Annotated[
            str | None, Header(alias="Idempotency-Key", max_length=128)
        ] = None,
    ) -> RunStatusResponse:
        """Queue a diagnosis for the independent persistent worker."""
        state = RunState(
            agent_mode=settings.agent_mode,
            prompt_version=(
                "multi-agent-graph-v7-contracts"
                if settings.agent_mode == "multi"
                else "single-agent-v12"
            ),
            incident=request.incident,
            memory_policy=settings.memory_policy,
            learning_policy=settings.memory_policy,
            knowledge_policy=settings.knowledge_policy,
            verification_policy=settings.verification_policy,
            investigation_start=request.investigation_start,
            investigation_end=request.investigation_end,
            limits=settings.run_limits,
            allowed_tools=settings.agent_allowed_tools,
            status=RunStatus.QUEUED,
        )
        request_hash = request_fingerprint(request)
        try:
            record = repository.create(
                state,
                request_hash=request_hash,
                idempotency_key=idempotency_key,
            )
        except RunConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        response.headers["Location"] = f"/api/v1/runs/{record.state.id}"
        return run_status(record)

    @application.get(
        "/api/v1/runs/{run_id}",
        response_model=RunStatusResponse,
        tags=["runs"],
    )
    def get_run(
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
    def get_run_result(
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
        "/api/v1/runs/{run_id}/cancel",
        status_code=202,
        response_model=RunStatusResponse,
        tags=["runs"],
    )
    def cancel_run(
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
        "/api/v1/runs/{run_id}/resume",
        status_code=202,
        response_model=RunStatusResponse,
        tags=["runs"],
    )
    def resume_run(
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
        "/api/v1/investigations",
        status_code=202,
        response_model=InvestigationCreated,
        tags=["investigations"],
    )
    def create_investigation(
        request: DiagnosisRequest,
        response: Response,
        settings: Annotated[Settings, Depends(get_request_settings)],
        repository: Annotated[RunRepository, Depends(get_run_repository)],
        idempotency_key: Annotated[
            str | None, Header(alias="Idempotency-Key", max_length=128)
        ] = None,
    ) -> InvestigationCreated:
        """Start autonomous diagnosis with optional later human follow-up."""
        state = RunState(
            agent_mode=settings.agent_mode,
            prompt_version=(
                "multi-agent-graph-v7-contracts"
                if settings.agent_mode == "multi"
                else "single-agent-v12"
            ),
            investigation_id=uuid4(),
            incident=request.incident,
            memory_policy=settings.memory_policy,
            learning_policy=settings.memory_policy,
            knowledge_policy=settings.knowledge_policy,
            verification_policy=settings.verification_policy,
            investigation_start=request.investigation_start,
            investigation_end=request.investigation_end,
            limits=settings.run_limits,
            allowed_tools=settings.agent_allowed_tools,
            status=RunStatus.QUEUED,
        )
        try:
            record = repository.create(
                state,
                request_hash=request_fingerprint(request, "investigation"),
                idempotency_key=idempotency_key,
                new_investigation=True,
            )
        except RunConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        response.headers["Location"] = (
            f"/api/v1/investigations/{record.state.investigation_id}"
        )
        return InvestigationCreated(
            id=record.state.investigation_id, run=run_status(record)
        )

    @application.get(
        "/api/v1/investigations/{investigation_id}",
        response_model=InvestigationResponse,
        tags=["investigations"],
    )
    def get_investigation(
        investigation_id: UUID,
        repository: Annotated[RunRepository, Depends(get_run_repository)],
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ) -> InvestigationResponse:
        """Read the preserved incident and chronological round history."""
        try:
            record = repository.get_investigation(
                investigation_id,
                offset=offset,
                limit=limit,
            )
        except RunNotFound as exc:
            raise HTTPException(
                status_code=404, detail="investigation not found"
            ) from exc
        return InvestigationResponse(
            id=record.id,
            incident=record.incident,
            created_at=record.created_at,
            next_offset=record.next_offset,
            turns=[
                InvestigationTurn(
                    run_id=s.id,
                    parent_run_id=s.parent_run_id,
                    purpose=s.purpose,
                    status=s.status,
                    stop_reason=s.stop_reason,
                    investigation_start=s.investigation_start,
                    investigation_end=s.investigation_end,
                    user_message=s.user_message,
                    diagnosis=s.diagnosis,
                )
                for s in record.runs
            ],
        )

    @application.post(
        "/api/v1/investigations/{investigation_id}/messages",
        status_code=202,
        response_model=RunStatusResponse,
        tags=["investigations"],
    )
    def follow_up(
        investigation_id: UUID,
        request: FollowUpRequest,
        response: Response,
        settings: Annotated[Settings, Depends(get_request_settings)],
        repository: Annotated[RunRepository, Depends(get_run_repository)],
        idempotency_key: Annotated[
            str | None, Header(alias="Idempotency-Key", max_length=128)
        ] = None,
    ) -> RunStatusResponse:
        """Queue a separately budgeted explanation or hypothesis check."""
        try:
            record = repository.follow_up(
                investigation_id,
                message=request.message,
                purpose=RunPurpose(request.purpose),
                limits=settings.run_limits,
                allowed_tools=settings.agent_allowed_tools,
                memory_policy=settings.memory_policy,
                start=request.investigation_start,
                end=request.investigation_end,
                request_hash=request_fingerprint(
                    request, str(investigation_id)
                ),
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
        run_id: UUID,
        request: Request,
        repository: Annotated[RunRepository, Depends(get_run_repository)],
        after: Annotated[int, Query(ge=0)] = 0,
        last_event_id: Annotated[
            str | None, Header(alias="Last-Event-ID")
        ] = None,
    ) -> StreamingResponse:
        """Subscribe to readable progress or replay from a durable cursor."""
        if last_event_id is not None:
            if (
                not last_event_id.isascii()
                or not last_event_id.isdigit()
                or len(last_event_id) > 20
            ):
                raise HTTPException(status_code=400, detail="invalid event ID")
            after = int(last_event_id)
        try:
            snapshot = await asyncio.to_thread(
                repository.event_snapshot,
                run_id,
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

    @application.post(
        "/api/v1/memories",
        status_code=201,
        response_model=MemoryRecord,
        tags=["memory"],
    )
    def confirm_memory(
        request: MemoryCreate,
        repository: Annotated[RunRepository, Depends(get_run_repository)],
    ) -> MemoryRecord:
        """Attest a resolution manually; no diagnostic tool writes memories."""
        try:
            return MemoryRepository(repository.database).confirm(request)
        except RunNotFound as exc:
            raise HTTPException(404, "source run not found") from exc
        except RunConflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @application.get(
        "/api/v1/memories", response_model=list[MemoryRecord], tags=["memory"]
    )
    def list_memories(
        repository: Annotated[RunRepository, Depends(get_run_repository)],
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ) -> list[MemoryRecord]:
        """Page confirmations, including expired and revoked entries."""
        return MemoryRepository(repository.database).list(
            offset=offset, limit=limit
        )

    @application.get(
        "/api/v1/memories/snapshot",
        response_model=MemorySnapshot,
        tags=["memory"],
    )
    def export_memory_snapshot(
        environment: Annotated[ScopeName, Query()],
        deployment_version: Annotated[ScopeName, Query()],
        service: Annotated[
            list[ScopeName], Query(min_length=1, max_length=20)
        ],
        repository: Annotated[RunRepository, Depends(get_run_repository)],
        limit: Annotated[int, Query(ge=1, le=10)] = 3,
    ) -> MemorySnapshot:
        """Freeze a scoped corpus before collecting held-out test incidents."""
        policy = MemoryPolicy(
            scope=MemoryScope(
                environment=environment,
                deployment_version=deployment_version,
            ),
            max_records=limit,
        )
        return MemoryRepository(repository.database).snapshot(
            policy,
            Incident(
                title="Memory export",
                description="Snapshot",
                affected_services=service,
            ),
            datetime.now(UTC),
        )

    @application.get(
        "/api/v1/memories/{memory_id}",
        response_model=MemoryRecord,
        tags=["memory"],
    )
    def get_memory(
        memory_id: UUID,
        repository: Annotated[RunRepository, Depends(get_run_repository)],
    ) -> MemoryRecord:
        """Inspect confirmation and provenance without agent retrieval."""
        try:
            return MemoryRepository(repository.database).get(memory_id)
        except MemoryNotFound as exc:
            raise HTTPException(404, "memory not found") from exc

    @application.put(
        "/api/v1/memories/{memory_id}",
        response_model=MemoryRecord,
        tags=["memory"],
    )
    def revise_memory(
        memory_id: UUID,
        request: MemoryRevision,
        repository: Annotated[RunRepository, Depends(get_run_repository)],
    ) -> MemoryRecord:
        """Reconfirm a correction with the revision the reviewer inspected."""
        try:
            return MemoryRepository(repository.database).confirm(
                request,
                memory_id=memory_id,
                expected_revision=request.expected_revision,
            )
        except (MemoryNotFound, RunNotFound) as exc:
            raise HTTPException(404, "memory or source run not found") from exc
        except RunConflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @application.post(
        "/api/v1/memories/{memory_id}/revoke",
        response_model=MemoryRecord,
        tags=["memory"],
    )
    def revoke_memory(
        memory_id: UUID,
        expected_revision: Annotated[int, Query(ge=1)],
        repository: Annotated[RunRepository, Depends(get_run_repository)],
    ) -> MemoryRecord:
        """Remove a disputed resolution from future retrieval."""
        try:
            return MemoryRepository(repository.database).revoke(
                memory_id,
                expected_revision=expected_revision,
            )
        except MemoryNotFound as exc:
            raise HTTPException(404, "memory not found") from exc
        except RunConflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @application.delete(
        "/api/v1/memories/{memory_id}", status_code=204, tags=["memory"]
    )
    def delete_memory(
        memory_id: UUID,
        expected_revision: Annotated[int, Query(ge=1)],
        repository: Annotated[RunRepository, Depends(get_run_repository)],
    ) -> Response:
        """Delete current content; prior run snapshots remain auditable."""
        try:
            MemoryRepository(repository.database).delete(
                memory_id,
                expected_revision=expected_revision,
            )
        except RunConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        return Response(status_code=204)

    @application.get(
        "/api/v1/guidance",
        response_model=list[GuidanceRecord],
        tags=["memory"],
    )
    def list_guidance(
        repository: Annotated[RunRepository, Depends(get_run_repository)],
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ) -> list[GuidanceRecord]:
        """Review the local user's inferred preferences and skills."""
        return MemoryRepository(repository.database).list_guidance(
            offset=offset, limit=limit
        )

    @application.put(
        "/api/v1/guidance/{guidance_id}",
        response_model=GuidanceRecord,
        tags=["memory"],
    )
    def revise_guidance(
        guidance_id: UUID,
        request: GuidanceRevision,
        repository: Annotated[RunRepository, Depends(get_run_repository)],
    ) -> GuidanceRecord:
        """Correct an inferred instruction with optimistic concurrency."""
        try:
            return MemoryRepository(repository.database).revise_guidance(
                guidance_id, request
            )
        except MemoryNotFound as exc:
            raise HTTPException(404, "guidance not found") from exc
        except RunConflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @application.delete(
        "/api/v1/guidance/{guidance_id}", status_code=204, tags=["memory"]
    )
    def delete_guidance(
        guidance_id: UUID,
        expected_revision: Annotated[int, Query(ge=1)],
        repository: Annotated[RunRepository, Depends(get_run_repository)],
    ) -> Response:
        """Remove an inferred instruction from future retrieval."""
        try:
            MemoryRepository(repository.database).delete_guidance(
                guidance_id, expected_revision=expected_revision
            )
        except RunConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        return Response(status_code=204)

    return application


app = create_app()
