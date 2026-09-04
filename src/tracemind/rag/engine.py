"""Embedded LightRAG lifecycle and provenance-aware hybrid retrieval."""

import asyncio
import fcntl
import json
import logging
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from importlib.metadata import version
from functools import partial
from math import isfinite
from time import monotonic

from tracemind.rag.config import RagSettings
from tracemind.async_utils import run_io
from tracemind.rag.fusion import fuse_candidates
from tracemind.rag.ingestion import snapshot_path, verify_snapshot
from tracemind.rag.keyword import KeywordIndex
from tracemind.rag.model_bridge import Completion, LightRAGModelBridge
from tracemind.rag.local_models import BgeTokenizer, LocalModels
from tracemind.rag.models import RetrievalHit

logger = logging.getLogger(__name__)
# Pinned LightRAG reads Qdrant configuration from process environment at init.
_qdrant_initialization_lock = asyncio.Lock()


class RagRuntime:
    """Own one local workspace and serialize operations on its file stores.

    Published knowledge is immutable through this API. The empty bootstrap
    workspace remains available for infrastructure checks.
    """

    def __init__(
        self,
        settings: RagSettings,
        *,
        completion: Completion | None = None,
        models: LocalModels | None = None,
    ) -> None:
        """Capture a fixed snapshot and defer external initialization."""
        self.settings = settings
        self.models = models or LocalModels(settings)
        self.model = LightRAGModelBridge(settings, completion)
        self.index: KeywordIndex | None = None
        self.rag = None
        self._lock = asyncio.Lock()
        self._workspace_lock = None
        self._published = False
        self._building = False
        self._reader_directory = None

    async def initialize(self, *, chunking_func=None) -> None:
        """Open Qdrant, NetworkX and local metadata with pinned models."""
        settings = self.settings
        if self.rag is not None:
            raise RuntimeError("RAG runtime is already initialized")
        self._published = snapshot_path(settings.workspace).exists()
        self._building = chunking_func is not None
        if self._published:
            verify_snapshot(settings.workspace)
            if self._building:
                raise ValueError(
                    "published knowledge snapshot cannot be modified"
                )
        elif (
            settings.workspace / "build.json"
        ).exists() and not self._building:
            raise ValueError(
                "knowledge build is incomplete and cannot be queried"
            )
        locks = settings.directory / "locks"
        locks.mkdir(parents=True, exist_ok=True)
        handle = (locks / f"{settings.snapshot}.lock").open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise RuntimeError(
                "another process owns this RAG workspace"
            ) from None
        self._workspace_lock = handle
        try:
            await self._initialize(chunking_func=chunking_func)
        except BaseException:
            await self.close()
            raise

    async def _initialize(self, *, chunking_func=None) -> None:
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc, Tokenizer

        settings = self.settings
        settings.workspace.mkdir(parents=True, exist_ok=True)
        manifest = settings.workspace / "runtime.json"
        identity = {
            "snapshot": settings.snapshot,
            "lightrag": version("lightrag-hku"),
            "qdrant_url": str(settings.qdrant_url),
            "embedding": settings.embedding_model,
            "embedding_revision": settings.embedding_revision,
            "dimension": settings.embedding_dimension,
            "max_tokens": settings.embedding_max_tokens,
            "reranker": settings.reranker_model,
            "reranker_revision": settings.reranker_revision,
            "chunk_tokens": settings.chunk_tokens,
            "llm_model": settings.llm_model,
            "corpus_published": False,
        }
        if manifest.exists():
            previous = json.loads(manifest.read_text())
            immutable = ("snapshot", "lightrag", "qdrant_url", "embedding",
                         "embedding_revision", "dimension", "max_tokens",
                         "chunk_tokens")
            if any(previous.get(key) != identity[key] for key in immutable):
                raise ValueError(
                    "workspace configuration changed; use a new ID")
        await run_io(self.models.load)
        self.index = KeywordIndex(
            settings.workspace / "keyword.sqlite3", readonly=self._published
        )
        working_dir = settings.workspace / "lightrag"
        if self._published:
            # Framework stores may flush caches/metadata during close. Never
            # point their writable lifetime at immutable publication artifacts.
            self._reader_directory = TemporaryDirectory(prefix="tracemind-rag-")
            reader = Path(self._reader_directory.name) / "lightrag"
            shutil.copytree(working_dir, reader)
            working_dir = reader
        async with _qdrant_initialization_lock:
            previous_url = os.environ.get("QDRANT_URL")
            os.environ["QDRANT_URL"] = str(settings.qdrant_url)
            try:
                special_tokens = (
                    self.models.embedding.tokenizer.num_special_tokens_to_add(
                        pair=False
                    )
                )
                self.rag = LightRAG(
                    working_dir=str(working_dir),
                    workspace=settings.snapshot,
                    vector_storage="QdrantVectorDBStorage",
                    graph_storage="NetworkXStorage",
                    kv_storage="JsonKVStorage",
                    doc_status_storage="JsonDocStatusStorage",
                    tokenizer=Tokenizer(
                        settings.embedding_model,
                        BgeTokenizer(self.models.embedding.tokenizer),
                    ),
                    embedding_func=EmbeddingFunc(
                        embedding_dim=settings.embedding_dimension,
                        max_token_size=(
                            settings.embedding_max_tokens
                            - special_tokens
                        ),
                        model_name=settings.embedding_model,
                        supports_asymmetric=True,
                        func=self.models.embed,
                    ),
                    llm_model_func=self.model.complete,
                    role_llm_configs={
                        "keyword": {"func": partial(
                            self.model.complete, keyword_extraction=True)},
                    },
                    llm_model_name=settings.llm_model or "injected-model",
                    llm_model_max_async=1,
                    embedding_func_max_async=1,
                    embedding_batch_num=settings.batch_size,
                    chunk_token_size=settings.chunk_tokens,
                    chunk_overlap_token_size=0,
                    summary_max_tokens=settings.summary_max_tokens,
                    summary_length_recommended=(
                        settings.summary_length_recommended),
                    max_parallel_insert=1,
                    enable_llm_cache=not self._published,
                    **({"chunking_func": chunking_func}
                       if chunking_func else {}),
                )
                await self.rag.initialize_storages()
                if not manifest.exists():
                    manifest.write_text(json.dumps(identity, indent=2) + "\n")
            except BaseException:
                await self.close()
                raise
            finally:
                if previous_url is None:
                    os.environ.pop("QDRANT_URL", None)
                else:
                    os.environ["QDRANT_URL"] = previous_url

    async def close(self) -> None:
        """Flush framework stores and release metadata even on errors."""
        try:
            if self.rag is not None:
                await self.rag.finalize_storages()
        finally:
            if self.index is not None:
                self.index.close()
                self.index = None
            self.rag = None
            if self._workspace_lock is not None:
                self._workspace_lock.close()
                self._workspace_lock = None
            if self._reader_directory is not None:
                self._reader_directory.cleanup()
                self._reader_directory = None

    async def require_empty_vectors(self) -> None:
        """Prevent reuse of a remote namespace from a different local
        checkout.
        """
        from lightrag.kg.qdrant_impl import workspace_filter_condition
        from qdrant_client import QdrantClient, models

        def check():
            """Use the public Qdrant API, independent of storage internals."""
            client = QdrantClient(url=str(self.settings.qdrant_url))
            try:
                for store in (self.rag.chunks_vdb, self.rag.entities_vdb,
                              self.rag.relationships_vdb):
                    count = client.count(
                        collection_name=store.final_namespace,
                        count_filter=models.Filter(must=[
                            workspace_filter_condition(
                                store.effective_workspace)]),
                        exact=True,
                    )
                    if count.count:
                        raise ValueError(
                            "Qdrant workspace already contains vectors; "
                            "use a new snapshot ID")
            finally:
                client.close()

        await run_io(check)

    async def search(
        self,
        query: str,
        *,
        mode: str | None = None,
        high_level_keywords: list[str] | None = None,
        low_level_keywords: list[str] | None = None,
        queries: list[str] | None = None,
    ) -> list[RetrievalHit]:
        """Retrieve structured context, fuse BM25, then rerank references.

        No final answer is generated by LightRAG. Its keyword extraction may
        still invoke the counted model bridge. Unregistered chunks fail closed.
        """
        from lightrag import QueryParam

        if self.rag is None or self.index is None:
            raise RuntimeError("initialize the RAG runtime first")
        selected_mode = mode or self.settings.mode
        if selected_mode not in {"mix", "naive"}:
            raise ValueError("only mix and naive retrieval are supported")
        if not query.strip():
            raise ValueError("retrieval query must not be blank")
        if self._building or (self.index.count and not self._published):
            raise ValueError(
                "retrieval requires a published knowledge snapshot"
            )
        async with self._lock, asyncio.timeout(self.settings.timeout_seconds):
            self.model.reset()
            if self.index.count == 0:
                return []
            started = monotonic()
            branches = {}
            for number, planned in enumerate(queries or [query]):
                result = await self.rag.aquery_data(
                    planned, QueryParam(
                        mode=selected_mode,
                        only_need_context=True,
                        enable_rerank=False,
                        top_k=self.settings.candidate_limit,
                        chunk_top_k=self.settings.candidate_limit,
                        hl_keywords=high_level_keywords or [],
                        ll_keywords=low_level_keywords or [],
                        max_total_tokens=(
                            self.settings.retrieval_context_tokens),
                    ),
                )
                no_results = (
                    result.get("metadata", {}).get("failure_reason")
                    == "no_results"
                )
                if result.get("status") != "success" and not no_results:
                    raise RuntimeError("LightRAG retrieval did not succeed")
                chunks = []
                for item in [] if no_results else result["data"]["chunks"]:
                    chunk = self.index.get(item["chunk_id"])
                    if item["content"] != chunk.text:
                        raise ValueError(
                            "retrieved text differs from canonical chunk")
                    chunks.append(chunk)
                suffix = f":{number}" if len(queries or [query]) > 1 else ""
                branches[selected_mode + suffix] = chunks
                if self.settings.bm25_enabled:
                    branches["bm25" + suffix] = self.index.search(
                        planned, limit=self.settings.candidate_limit)
            hits = fuse_candidates(
                branches, limit=self.settings.candidate_limit
            )
            scores = await self.models.rank(
                query, [h.chunk.text for h in hits]
            )
            if len(scores) != len(hits) or not all(map(isfinite, scores)):
                raise ValueError("reranker returned invalid scores")
            for hit, score in zip(hits, scores):
                hit.rerank_score = score
            hits.sort(
                key=lambda h: (
                    -h.rerank_score,
                    -h.fusion_score,
                    h.chunk.chunk_id,
                )
            )
            for hit in hits[:self.settings.result_limit]:
                if hit.chunk.parent_id is not None:
                    hit.parent = self.index.get_parent(hit.chunk.parent_id)
            logger.info(
                "rag.search snapshot=%s mode=%s candidates=%s "
                "model_calls=%s input_tokens=%s output_tokens=%s "
                "unreported_calls=%s elapsed_ms=%.1f",
                self.settings.snapshot,
                selected_mode,
                len(hits),
                self.model.calls,
                self.model.input_tokens,
                self.model.output_tokens,
                self.model.unreported_calls,
                (monotonic() - started) * 1000,
            )
            return hits[: self.settings.result_limit]
