"""Async network I/O with the pinned LightRAG Qdrant storage format."""

import asyncio
import os
from contextlib import closing
from dataclasses import dataclass
from time import time

import numpy as np
from lightrag.base import BaseVectorStorage
from lightrag.kg import (
    STORAGES, STORAGE_ENV_REQUIREMENTS, STORAGE_IMPLEMENTATIONS,
)
from lightrag.kg.qdrant_impl import (
    CREATED_AT_FIELD, DEFAULT_WORKSPACE, ENTITY_PREFIX, ID_FIELD,
    WORKSPACE_ID_FIELD, QdrantVectorDBStorage, _PendingVectorDoc,
    DEFAULT_QDRANT_DELETE_MAX_POINTS_PER_BATCH,
    DEFAULT_QDRANT_UPSERT_MAX_PAYLOAD_BYTES,
    DEFAULT_QDRANT_UPSERT_MAX_POINTS_PER_BATCH,
    compute_mdhash_id_for_qdrant, config, workspace_filter_condition,
)
from lightrag.kg.shared_storage import get_namespace_lock
from lightrag.utils import compute_mdhash_id, validate_workspace
from qdrant_client import AsyncQdrantClient, QdrantClient, models

from tracemind.async_utils import run_io


@dataclass
class AsyncQdrantVectorDBStorage(BaseVectorStorage):
    """Retain canonical IDs, payloads and buffered writes for ingestion/query.

    Collection setup reuses the pinned migration implementation in a thread;
    runtime requests use an async client. Snapshot deletion is not supported.
    """

    def __post_init__(self):
        validate_workspace(self.workspace)
        self._validate_embedding_func()
        self.effective_workspace = self.workspace or DEFAULT_WORKSPACE
        self.model_suffix = self._generate_collection_suffix()
        self.final_namespace = f"lightrag_vdb_{self.namespace}"
        if self.model_suffix:
            self.final_namespace += "_" + self.model_suffix
        options = self.global_config["vector_db_storage_cls_kwargs"]
        self.cosine_better_than_threshold = options[
            "cosine_better_than_threshold"]
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._max_upsert_payload_bytes = int(os.environ.get(
            "QDRANT_UPSERT_MAX_PAYLOAD_BYTES",
            DEFAULT_QDRANT_UPSERT_MAX_PAYLOAD_BYTES))
        self._max_upsert_points_per_batch = int(os.environ.get(
            "QDRANT_UPSERT_MAX_POINTS_PER_BATCH",
            DEFAULT_QDRANT_UPSERT_MAX_POINTS_PER_BATCH))
        self._max_delete_points_per_batch = int(os.environ.get(
            "QDRANT_DELETE_MAX_POINTS_PER_BATCH",
            DEFAULT_QDRANT_DELETE_MAX_POINTS_PER_BATCH))
        self._pending_vector_docs: dict[str, _PendingVectorDoc] = {}
        self._pending_vector_deletes: set[str] = set()
        self._flush_lock = None
        self._client = None
        self._initialized = False

    async def initialize(self):
        """Set up the collection before exposing its async transport."""
        if self._initialized:
            return
        options = self.global_config["vector_db_storage_cls_kwargs"]
        url = options["qdrant_url"]
        timeout = options["qdrant_timeout"]
        api_key = os.environ.get(
            "QDRANT_API_KEY", config.get("qdrant", "apikey", fallback=None))

        def setup():
            with closing(QdrantClient(
                url=url, api_key=api_key, timeout=timeout,
            )) as client:
                QdrantVectorDBStorage.setup_collection(
                    client, self.final_namespace, namespace=self.namespace,
                    workspace=self.effective_workspace,
                    vectors_config=models.VectorParams(
                        size=self.embedding_func.embedding_dim,
                        distance=models.Distance.COSINE),
                    hnsw_config=models.HnswConfigDiff(payload_m=16, m=0),
                    model_suffix=self.model_suffix,
                )

        await run_io(setup)
        self._client = AsyncQdrantClient(
            url=url, api_key=api_key, timeout=timeout,
            check_compatibility=False)
        self._flush_lock = get_namespace_lock(
            namespace=self.final_namespace, workspace=self.effective_workspace)
        self._initialized = True

    async def upsert(self, data):
        """Buffer payloads using LightRAG's canonical Qdrant field names."""
        created = int(time())
        async with self._flush_lock:
            for doc_id, value in data.items():
                source = {ID_FIELD: doc_id,
                          WORKSPACE_ID_FIELD: self.effective_workspace,
                          CREATED_AT_FIELD: created,
                          **{key: item for key, item in value.items()
                             if key in self.meta_fields}}
                self._pending_vector_deletes.discard(doc_id)
                self._pending_vector_docs[doc_id] = _PendingVectorDoc(
                    source=source, content=value["content"])

    async def delete(self, ids):
        """Buffer tombstones until the next committed flush."""
        async with self._flush_lock:
            for doc_id in ids:
                self._pending_vector_docs.pop(doc_id, None)
                self._pending_vector_deletes.add(doc_id)

    async def delete_entity(self, entity_name):
        """Use the pinned storage's entity identity."""
        await self.delete([
            compute_mdhash_id(entity_name, prefix=ENTITY_PREFIX)])

    async def drop_pending_index_ops(self):
        """Discard uncommitted buffers when the ingestion pipeline aborts."""
        async with self._flush_lock:
            self._pending_vector_docs.clear()
            self._pending_vector_deletes.clear()

    async def index_done_callback(self):
        """Commit the pipeline's buffered vector operations."""
        await self._flush_pending_vector_ops()

    async def query(self, query, top_k, query_embedding=None):
        """Await remote recall without blocking the investigation loop."""
        if query_embedding is None:
            vectors = await self.embedding_func(
                [query], context="query", _priority=5)
            query_embedding = vectors[0]
        result = await self._client.query_points(
            collection_name=self.final_namespace, query=query_embedding,
            limit=top_k, with_payload=True,
            score_threshold=self.cosine_better_than_threshold,
            query_filter=models.Filter(must=[
                workspace_filter_condition(self.effective_workspace)]),
        )
        return [{**point.payload, "distance": point.score,
                 CREATED_AT_FIELD: point.payload.get(CREATED_AT_FIELD)}
                for point in result.points]

    async def _embed_pending(self, docs):
        """Cache embeddings so a failed remote flush can retry unchanged."""
        missing = [doc for doc in docs if doc.vector is None]
        if not missing:
            return
        batches = [missing[i:i + self._max_batch_size]
                   for i in range(0, len(missing), self._max_batch_size)]
        vectors = np.concatenate(await asyncio.gather(*[
            self.embedding_func([doc.content for doc in batch],
                                context="document")
            for batch in batches]))
        if len(vectors) != len(missing):
            raise ValueError("embedding count differs from pending vectors")
        for doc, vector in zip(missing, vectors):
            doc.vector = np.asarray(vector, dtype=np.float32).tolist()

    async def _flush_pending_vector_ops(self):
        """Commit buffers only after every remote batch succeeds."""
        async with self._flush_lock:
            docs = self._pending_vector_docs
            deletes = self._pending_vector_deletes
            await self._embed_pending(list(docs.values()))
            points = [models.PointStruct(
                id=compute_mdhash_id_for_qdrant(
                    doc_id, prefix=self.effective_workspace),
                vector=doc.vector, payload=dict(doc.source))
                for doc_id, doc in docs.items()]
            for batch, _ in QdrantVectorDBStorage._build_upsert_batches(
                points, self._max_upsert_payload_bytes,
                self._max_upsert_points_per_batch,
            ):
                await self._client.upsert(
                    collection_name=self.final_namespace,
                    points=batch, wait=True)
            ids = [compute_mdhash_id_for_qdrant(
                doc_id, prefix=self.effective_workspace) for doc_id in deletes]
            batch_size = (self._max_delete_points_per_batch
                          if self._max_delete_points_per_batch > 0
                          else len(ids) or 1)
            for i in range(0, len(ids), batch_size):
                await self._client.delete(
                    collection_name=self.final_namespace,
                    points_selector=models.PointIdsList(
                        points=ids[i:i + batch_size]), wait=True)
            docs.clear()
            deletes.clear()

    async def get_by_id(self, id):
        """Preserve the same ordered read-your-writes path for one ID."""
        return (await self.get_by_ids([id]))[0]

    async def get_by_ids(self, ids):
        """Return payloads in request order, including missing IDs."""
        buffered, remaining = {}, []
        async with self._flush_lock:
            for doc_id in ids:
                if doc_id in self._pending_vector_deletes:
                    buffered[doc_id] = None
                elif doc_id in self._pending_vector_docs:
                    doc = self._pending_vector_docs[doc_id]
                    buffered[doc_id] = {
                        CREATED_AT_FIELD: None, **doc.source}
                else:
                    remaining.append(doc_id)
        remote = {}
        if remaining:
            points = await self._client.retrieve(
                collection_name=self.final_namespace,
                ids=[compute_mdhash_id_for_qdrant(
                    doc_id, prefix=self.effective_workspace)
                    for doc_id in remaining], with_payload=True)
            for point in points:
                payload = {CREATED_AT_FIELD: None, **(point.payload or {})}
                remote[point.id.replace("-", "")] = payload
        return [buffered[doc_id] if doc_id in buffered else remote.get(
            compute_mdhash_id_for_qdrant(
                doc_id, prefix=self.effective_workspace)) for doc_id in ids]

    async def get_vectors_by_ids(self, ids):
        """Read pending/stored vectors and propagate backend errors."""
        result, remaining = {}, []
        async with self._flush_lock:
            pending = {doc_id: self._pending_vector_docs[doc_id]
                       for doc_id in ids
                       if doc_id in self._pending_vector_docs
                       and doc_id not in self._pending_vector_deletes}
            await self._embed_pending(list(pending.values()))
            for doc_id in ids:
                if doc_id in self._pending_vector_deletes:
                    continue
                if doc_id in pending:
                    result[doc_id] = pending[doc_id].vector
                else:
                    remaining.append(doc_id)
        if remaining:
            requested = {compute_mdhash_id_for_qdrant(
                doc_id, prefix=self.effective_workspace): doc_id
                for doc_id in remaining}
            points = await self._client.retrieve(
                collection_name=self.final_namespace, ids=list(requested),
                with_vectors=True, with_payload=False)
            result.update({requested[point.id.replace("-", "")]: point.vector
                           for point in points if point.vector is not None})
        return result

    async def finalize(self):
        """Close the transport on flush failure as well as success."""
        if self._client is not None:
            try:
                await self._flush_pending_vector_ops()
            finally:
                await self._client.close()
                self._client = None

    async def delete_entity_relation(self, entity_name):
        """The snapshot runtime does not expose graph editing."""
        raise NotImplementedError("snapshot entity deletion is not supported")

    async def drop(self):
        """Snapshot lifecycle is managed outside the retrieval runtime."""
        raise NotImplementedError("snapshot deletion is not supported")


def register_storage():
    """Register the application adapter through LightRAG's storage registry."""
    name = AsyncQdrantVectorDBStorage.__name__
    STORAGES[name] = __name__
    implementations = STORAGE_IMPLEMENTATIONS[
        "VECTOR_STORAGE"]["implementations"]
    if name not in implementations:
        implementations.append(name)
    STORAGE_ENV_REQUIREMENTS[name] = []
