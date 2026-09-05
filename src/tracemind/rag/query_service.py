"""Prewarmed retrieval process without provider credentials."""
import asyncio
from contextlib import asynccontextmanager
import logging
from time import monotonic

from fastapi import FastAPI, HTTPException

from tracemind.async_utils import run_io
from tracemind.rag.config import RagSettings
from tracemind.rag.engine import RagRuntime
from tracemind.rag.ingestion import verify_snapshot
from tracemind.rag.query_contracts import RetrievalRequest, RetrievalResponse

logger = logging.getLogger(__name__)


def create_app(settings: RagSettings | None = None, *, runtime=None):
    settings = settings or RagSettings()
    owner = runtime or RagRuntime(settings)
    admission = asyncio.Lock()
    ready = False

    @asynccontextmanager
    async def lifespan(app):
        nonlocal ready
        started = monotonic()
        try:
            manifest = await run_io(verify_snapshot, settings.workspace)
            if manifest.get('snapshot') != settings.snapshot:
                raise ValueError('published snapshot identity mismatch')
            await owner.initialize()
            await owner.models.embed(['retrieval readiness'], context='query')
            await owner.models.rank('retrieval readiness', ['retrieval ready'])
            ready = True
            logger.info('rag.service.ready snapshot=%s startup_ms=%.1f',
                        settings.snapshot, (monotonic() - started) * 1000)
            yield
        finally:
            ready = False
            await owner.close()

    app = FastAPI(title='TraceMind retrieval', lifespan=lifespan)

    @app.get('/ready')
    async def readiness():
        if not ready:
            raise HTTPException(503, 'retrieval is not ready')
        return {'snapshot': settings.snapshot, 'ready': True}

    @app.post('/search', response_model=RetrievalResponse)
    async def search(request: RetrievalRequest):
        if not ready:
            raise HTTPException(503, 'retrieval is not ready')
        if request.snapshot != settings.snapshot:
            raise HTTPException(409, 'knowledge snapshot mismatch')
        # Reject overload without consuming the caller's deadline in a queue.
        if admission.locked():
            raise HTTPException(503, 'retrieval is busy')
        async with admission:
            try:
                async with asyncio.timeout(settings.service_query_timeout):
                    hits = await owner.search(
                        request.question,
                        queries=[q.query for q in request.queries],
                        query_keywords=[(
                            list(q.high_level_keywords),
                            list(q.low_level_keywords),
                        ) for q in request.queries],
                    )
            except TimeoutError as exc:
                raise HTTPException(
                    504, 'retrieval deadline exceeded') from exc
            return RetrievalResponse(snapshot=settings.snapshot, hits=hits)

    return app
