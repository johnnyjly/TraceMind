"""Retrieve published snapshots with an isolated model callback per search."""

import asyncio
import sqlite3
import logging
from functools import lru_cache
from time import monotonic

from qdrant_client.http.exceptions import ApiException

from tracemind.async_utils import run_io
from tracemind.rag.config import RagSettings
from tracemind.rag.engine import RagRuntime
from tracemind.rag.errors import RetrievalUnavailable
from tracemind.rag.ingestion import verify_snapshot
from tracemind.rag.model_bridge import Completion
from tracemind.rag.local_models import LocalModels
from tracemind.rag.query_contracts import RetrievalResponse


logger = logging.getLogger(__name__)


class SnapshotRetriever:
    """Share local weights while serializing this process's query runtimes.

    Each operation receives its own completion callback and closes all stores.
    The runtime's process lock remains authoritative across worker processes.
    """

    def __init__(self, settings: RagSettings) -> None:
        """Capture model configuration without loading weights or stores."""
        self.settings = settings
        self.models = LocalModels(settings)
        self._lock = asyncio.Lock()

    async def __call__(
        self,
        queries: list[str],
        *,
        question: str,
        snapshot: str,
        completion: Completion,
    ) -> RetrievalResponse:
        """Fuse every planned query, rerank once, and copy complete parents."""
        settings = self.settings.model_copy(update={"snapshot": snapshot})
        started = monotonic()
        async with self._lock:
            acquired = monotonic()
            try:
                manifest = await run_io(verify_snapshot, settings.workspace)
                if manifest.get("snapshot") != snapshot:
                    raise ValueError("published snapshot identity mismatch")
                runtime = RagRuntime(
                    settings, completion=completion, models=self.models
                )
                verified = monotonic()
                try:
                    await runtime.initialize()
                    logger.info(
                        "rag.ready snapshot=%s wait_ms=%.1f verify_ms=%.1f "
                        "initialize_ms=%.1f",
                        snapshot,
                        (acquired - started) * 1000,
                        (verified - acquired) * 1000,
                        (monotonic() - verified) * 1000,
                    )
                    hits = await runtime.search(question, queries=queries)
                    return RetrievalResponse(
                        snapshot=snapshot,
                        hits=hits,
                        diagnostics=runtime.last_retrieval,
                    )
                finally:
                    await runtime.close()
            except (ApiException, sqlite3.Error) as exc:
                raise RetrievalUnavailable(
                    "knowledge storage unavailable"
                ) from exc


@lru_cache(maxsize=4)
def get_snapshot_retriever(settings_json: str) -> SnapshotRetriever:
    """Cache weights and admission independently of investigation callbacks."""
    return SnapshotRetriever(RagSettings.model_validate_json(settings_json))
