"""Prewarmed retrieval process without provider credentials."""

import asyncio
from contextlib import asynccontextmanager
import logging
import re
from time import monotonic

from fastapi import FastAPI, HTTPException

from tracemind.async_utils import run_io
from tracemind.observability.langfuse import observation
from tracemind.rag.config import RagSettings
from tracemind.rag.engine import RagRuntime
from tracemind.rag.errors import RetrievalInputTooLong
from tracemind.rag.ingestion import verify_snapshot
from tracemind.rag.query_contracts import RetrievalRequest, RetrievalResponse

logger = logging.getLogger(__name__)


def create_app(
    settings: RagSettings | None = None, *, runtime=None, langfuse_client=None
):
    settings = settings or RagSettings()
    owner = runtime or RagRuntime(settings, langfuse_client=langfuse_client)
    admission = asyncio.Lock()
    ready = False
    snapshot_fingerprint = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal ready, snapshot_fingerprint
        started = monotonic()
        try:
            manifest = await run_io(verify_snapshot, settings.workspace)
            if manifest.get("snapshot") != settings.snapshot:
                raise ValueError("published snapshot identity mismatch")
            from hashlib import sha256
            import json

            snapshot_fingerprint = sha256(
                json.dumps(manifest, sort_keys=True).encode()
            ).hexdigest()
            await owner.initialize()
            await owner.models.embed(["retrieval readiness"], context="query")
            await owner.models.rank("retrieval readiness", ["retrieval ready"])
            ready = True
            logger.info(
                "rag.service.ready snapshot=%s startup_ms=%.1f",
                settings.snapshot,
                (monotonic() - started) * 1000,
            )
            yield
        finally:
            ready = False
            await owner.close()

    app = FastAPI(title="TraceMind retrieval", lifespan=lifespan)

    @app.middleware("http")
    async def trace_request(request, call_next):
        if request.url.path != "/search":
            return await call_next(request)
        parent = request.headers.get("traceparent", "")
        match = re.fullmatch(
            r"00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}", parent
        )
        context = (
            {"trace_id": match[1], "parent_span_id": match[2]}
            if match and int(match[1], 16) and int(match[2], 16)
            else None
        )
        with observation(
            langfuse_client, "knowledge.http", trace_context=context
        ) as span:
            response = await call_next(request)
            if span is not None:
                span.update(
                    metadata={"http_status": response.status_code},
                    level="ERROR"
                    if response.status_code >= 400
                    else "DEFAULT",
                )
            return response

    @app.get("/ready")
    async def readiness():
        if not ready:
            raise HTTPException(503, "retrieval is not ready")
        return {
            "snapshot": settings.snapshot,
            "ready": True,
            "snapshot_fingerprint": snapshot_fingerprint,
            "policy": owner.retrieval_policy(),
        }

    @app.post("/search", response_model=RetrievalResponse)
    async def search(request: RetrievalRequest):
        if not ready:
            raise HTTPException(503, "retrieval is not ready")
        if request.snapshot != settings.snapshot:
            raise HTTPException(409, "knowledge snapshot mismatch")
        # Reject overload without consuming the caller's deadline in a queue.
        if admission.locked():
            raise HTTPException(503, "retrieval is busy")
        async with admission:
            try:
                async with asyncio.timeout(settings.service_query_timeout):
                    hits = await owner.search(
                        request.question,
                        mode=request.mode,
                        queries=[q.query for q in request.queries],
                        query_keywords=[
                            (
                                list(q.high_level_keywords),
                                list(q.low_level_keywords),
                            )
                            for q in request.queries
                        ],
                    )
            except RetrievalInputTooLong as exc:
                raise HTTPException(422, str(exc)) from exc
            except TimeoutError as exc:
                raise HTTPException(
                    504, "retrieval deadline exceeded"
                ) from exc
            return RetrievalResponse(
                snapshot=settings.snapshot,
                hits=hits,
                diagnostics={
                    **{
                        k: v
                        for k, v in owner.last_retrieval.items()
                        if k not in {"query", "stages"}
                    },
                    "snapshot_fingerprint": snapshot_fingerprint,
                },
            )

    return app
