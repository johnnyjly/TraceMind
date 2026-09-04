"""Retrieve published snapshots with an isolated model callback per search."""

import asyncio
from functools import lru_cache

from tracemind.rag.config import RagSettings
from tracemind.rag.engine import RagRuntime
from tracemind.rag.ingestion import verify_snapshot
from tracemind.rag.model_bridge import Completion
from tracemind.rag.local_models import LocalModels
from tracemind.rag.models import RetrievalHit


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
        self, queries: list[str], *, question: str, snapshot: str,
        completion: Completion,
    ) -> list[RetrievalHit]:
        """Fuse every planned query, rerank once, and copy complete parents."""
        settings = self.settings.model_copy(update={"snapshot": snapshot})
        async with self._lock:
            manifest = await asyncio.to_thread(verify_snapshot,
                                               settings.workspace)
            if manifest.get("snapshot") != snapshot:
                raise ValueError("published snapshot identity mismatch")
            runtime = RagRuntime(settings, completion=completion,
                                 models=self.models)
            try:
                await runtime.initialize()
                return await runtime.search(question, queries=queries)
            finally:
                await runtime.close()


@lru_cache(maxsize=4)
def get_snapshot_retriever(settings_json: str) -> SnapshotRetriever:
    """Cache weights and admission independently of investigation callbacks."""
    return SnapshotRetriever(
        RagSettings.model_validate_json(settings_json))
