"""Exercise async Qdrant transport, storage compatibility and failed flushes."""

# Optional imports must be checked before importing retrieval modules.
# flake8: noqa: E402
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("lightrag", reason="requires the rag extra")
pytest.importorskip("qdrant_client", reason="requires the rag extra")

import numpy as np
from lightrag.utils import EmbeddingFunc
from lightrag.kg.qdrant_impl import compute_mdhash_id_for_qdrant
from qdrant_client import AsyncQdrantClient, QdrantClient, models

from tracemind.rag.qdrant import AsyncQdrantVectorDBStorage

pytestmark = pytest.mark.rag


def storage():
    calls = []

    async def embed(texts, **kwargs):
        calls.extend(texts)
        return np.array([[1., 0.] for text in texts])

    result = AsyncQdrantVectorDBStorage(
        namespace="chunks", workspace="adapter-test",
        embedding_func=EmbeddingFunc(
            embedding_dim=2, max_token_size=512, model_name="fixture", func=embed),
        meta_fields={"content", "full_doc_id"},
        global_config={"embedding_batch_num": 2,
                       "vector_db_storage_cls_kwargs": {
                           "cosine_better_than_threshold": .1,
                           "qdrant_url": "http://127.0.0.1:6333",
                           "qdrant_timeout": 1}},
    )
    result._flush_lock = asyncio.Lock()
    return result, calls


def test_query_transport_remains_cancellable_and_heartbeat_runs():
    async def exercise():
        entered, heartbeat, cancelled = [asyncio.Event() for _ in range(3)]

        async def handler(request):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def tick():
            await entered.wait()
            heartbeat.set()

        store, _ = storage()
        store._client = AsyncQdrantClient(
            url="http://qdrant.test", check_compatibility=False,
            transport=httpx.MockTransport(handler))
        ticking = asyncio.create_task(tick())
        try:
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(.05):
                    await store.query("RPC timeout", 3, query_embedding=[1., 0.])
            assert heartbeat.is_set() and cancelled.is_set()
        finally:
            await ticking
            await store.finalize()

    asyncio.run(exercise())


def test_flush_retry_preserves_ids_payload_order_and_cached_embeddings():
    async def exercise():
        store, embeddings = storage()
        client = AsyncMock()
        client.upsert.side_effect = [httpx.ConnectError("offline"), None]
        store._client = client
        await store.upsert({"a": {"content": "alpha", "full_doc_id": "doc-a"},
                            "b": {"content": "beta", "full_doc_id": "doc-b"}})
        with pytest.raises(httpx.ConnectError):
            await store.index_done_callback()
        assert set(store._pending_vector_docs) == {"a", "b"}
        assert embeddings == ["alpha", "beta"]
        await store.index_done_callback()
        assert embeddings == ["alpha", "beta"]
        assert not store._pending_vector_docs
        points = client.upsert.call_args.kwargs["points"]
        assert points[0].id == compute_mdhash_id_for_qdrant(
            "a", prefix=store.effective_workspace)
        assert points[0].payload["id"] == "a"
        assert points[0].payload["workspace_id"] == store.effective_workspace
        client.retrieve.return_value = [
            SimpleNamespace(id=p.id, payload=p.payload, vector=p.vector)
            for p in reversed(points)]
        results = await store.get_by_ids(["a", "missing", "b"])
        assert [row["id"] if row else None for row in results] == ["a", None, "b"]
        assert (await store.get_vectors_by_ids(["a", "b"])) == {
            "a": [1., 0.], "b": [1., 0.]}
        await store.delete(["a"])
        assert await store.get_by_id("a") is None
        await store.index_done_callback()
        client.delete.assert_awaited_once()
        await store.finalize()
        client.close.assert_awaited_once()

    asyncio.run(exercise())


def test_finalize_closes_transport_when_flush_fails():
    async def exercise():
        store, _ = storage()
        client = AsyncMock()
        client.upsert.side_effect = httpx.ConnectError("offline")
        store._client = client
        await store.upsert({"a": {"content": "alpha"}})
        with pytest.raises(httpx.ConnectError):
            await store.finalize()
        assert store._pending_vector_docs
        client.close.assert_awaited_once()
        assert store._client is None

    asyncio.run(exercise())


def test_async_adapter_reads_and_writes_canonical_sync_storage(tmp_path):
    store, _ = storage()
    location = str(tmp_path / "qdrant")
    legacy = QdrantClient(path=location)
    legacy.create_collection(
        store.final_namespace,
        vectors_config=models.VectorParams(
            size=2, distance=models.Distance.COSINE))
    legacy.upsert(store.final_namespace, [models.PointStruct(
        id=compute_mdhash_id_for_qdrant("old", prefix=store.workspace),
        vector=[1., 0.], payload={"id": "old", "content": "existing",
                                 "workspace_id": store.workspace})])
    legacy.close()

    async def exercise():
        store._client = AsyncQdrantClient(path=location)
        try:
            assert (await store.get_by_id("old"))["content"] == "existing"
            hits = await store.query("existing", 2, query_embedding=[1., 0.])
            assert [hit["id"] for hit in hits] == ["old"]
            await store.upsert({"new": {"content": "inserted"}})
            await store.index_done_callback()
        finally:
            await store.finalize()

    asyncio.run(exercise())
    legacy = QdrantClient(path=location)
    try:
        points = legacy.retrieve(store.final_namespace, [
            compute_mdhash_id_for_qdrant("new", prefix=store.workspace)])
        assert points[0].payload["content"] == "inserted"
        assert points[0].payload["workspace_id"] == store.workspace
    finally:
        legacy.close()


def test_collection_setup_drains_its_thread_without_blocking_loop(monkeypatch):
    import tracemind.rag.qdrant as adapter

    entered, release = threading.Event(), threading.Event()
    closed = []

    def setup(*args, **kwargs):
        entered.set()
        assert release.wait(3)

    monkeypatch.setattr(adapter, "QdrantClient", lambda **kwargs:
                        SimpleNamespace(close=lambda: closed.append(True)))
    monkeypatch.setattr(adapter.QdrantVectorDBStorage, "setup_collection", setup)

    async def exercise():
        store, _ = storage()
        task = asyncio.create_task(store.initialize())
        await asyncio.to_thread(entered.wait)
        task.cancel()
        await asyncio.sleep(.01)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert store._client is None

    try:
        asyncio.run(exercise())
    finally:
        release.set()
    assert closed == [True]
