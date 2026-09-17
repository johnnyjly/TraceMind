"""Check runtime ownership, batched ranking and independent IR evaluation."""
# Optional dependencies are checked before retrieval imports.
# flake8: noqa: E402

import asyncio
import sqlite3
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("ir_measures", reason="requires the rag extra")
pytest.importorskip("qdrant_client", reason="requires the rag extra")
pytest.importorskip("langchain_classic", reason="requires the rag extra")
pytest.importorskip("lightrag", reason="requires the rag extra")
from langchain_core.messages import AIMessage

from tracemind.evaluation.retrieval import RetrievalSuite, evaluate_retrieval
from tracemind.rag.config import RagSettings
from tracemind.rag.engine import RagRuntime
from tracemind.rag.keyword import KeywordIndex
from tracemind.rag.models import RetrievalHit
from tracemind.rag.retriever import SnapshotRetriever
from test_rag import chunk, settings

pytestmark = pytest.mark.rag


def test_real_framework_embedding_keeps_the_open_runtime_owner(
    tmp_path,
    monkeypatch,
):
    from lightrag import LightRAG
    import numpy as np
    from unittest.mock import AsyncMock

    class Tokens:
        def num_special_tokens_to_add(self, **kwargs):
            return 2

        def encode(self, text, **kwargs):
            return list(text.encode())

        def decode(self, tokens, **kwargs):
            return bytes(tokens).decode()

    received = []

    async def embed(texts, **kwargs):
        received.append((texts, kwargs))
        return np.zeros((len(texts), 2))

    # Keep the real constructor/config copying, without weights or Qdrant IO.
    monkeypatch.setattr(LightRAG, "initialize_storages", AsyncMock())
    monkeypatch.setattr(LightRAG, "finalize_storages", AsyncMock())
    models = SimpleNamespace(
        load=lambda: None,
        embedding=SimpleNamespace(tokenizer=Tokens()),
        embed=embed,
    )
    runtime = RagRuntime(
        settings(
            directory=tmp_path,
            snapshot="embedding-owner",
            embedding_dimension=2,
        ),
        models=models,
    )

    async def exercise():
        try:
            await runtime.initialize()
            assert runtime.index.count == 0
            vectors = await runtime.rag.embedding_func(
                ["request timeout"], context="query"
            )
            assert vectors.shape == (1, 2)
            assert received == [(["request timeout"], {"context": "query"})]
            assert runtime.index.count == 0
        finally:
            await runtime.close()
        assert runtime.index is None
        assert runtime.rag is None

    asyncio.run(exercise())


def test_planned_queries_fuse_before_exactly_one_rerank(tmp_path):
    runtime = RagRuntime(settings(result_limit=2))
    runtime._published = True
    runtime.index = KeywordIndex(tmp_path / "keyword.sqlite3")
    a, b = chunk("a", "RPC timeout"), chunk("b", "Queue saturation")
    runtime.index.register([a, b])
    queries, ranks = [], []

    async def retrieve(query, param):
        queries.append(query)
        item = a if query == "RPC timeout" else b
        return {
            "status": "success",
            "data": {
                "chunks": [{"chunk_id": item.chunk_id, "content": item.text}]
            },
        }

    async def rank(query, texts):
        ranks.append((query, texts))
        return [3 if text == b.text else 1 for text in texts]

    async def close():
        pass

    runtime.rag = SimpleNamespace(
        aquery_data=retrieve, finalize_storages=close
    )
    runtime.models.rank = rank

    async def exercise():
        try:
            hits = await runtime.search(
                "slow dependency", queries=["RPC timeout", "Queue saturation"]
            )
            assert [h.chunk.chunk_id for h in hits] == ["b", "a"]
        finally:
            await runtime.close()

    asyncio.run(exercise())
    assert queries == ["RPC timeout", "Queue saturation"]
    assert len(ranks) == 1
    assert ranks[0][0] == "slow dependency"
    assert set(ranks[0][1]) == {a.text, b.text}


def test_retriever_isolates_callbacks_and_closes_on_cancellation(monkeypatch):
    import tracemind.rag.retriever as retriever

    runtimes, received = [], []
    entered = None

    class Runtime:
        last_retrieval = {}

        def __init__(self, settings, *, completion, models):
            self.completion, self.models = completion, models
            self.closed = False
            runtimes.append(self)

        async def initialize(self):
            pass

        async def search(self, question, *, queries):
            received.append(await self.completion([]))
            if question == "cancel":
                entered.set()
                await asyncio.Event().wait()
            return []

        async def close(self):
            self.closed = True

    monkeypatch.setattr(retriever, "RagRuntime", Runtime)
    monkeypatch.setattr(
        retriever, "verify_snapshot", lambda path: {"snapshot": path.name}
    )
    search = SnapshotRetriever(RagSettings(_env_file=None))

    async def first(messages):
        return AIMessage(content="first-run")

    async def second(messages):
        return AIMessage(content="second-run")

    async def exercise():
        nonlocal entered
        entered = asyncio.Event()
        await asyncio.gather(
            search(["one"], question="one", snapshot="v1", completion=first),
            search(["two"], question="two", snapshot="v1", completion=second),
        )
        task = asyncio.create_task(
            search(
                ["three"], question="cancel", snapshot="v1", completion=first
            )
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert [m.content for m in received] == [
        "first-run",
        "second-run",
        "first-run",
    ]
    assert all(r.closed for r in runtimes)
    assert len({id(r.models) for r in runtimes}) == 1
    assert runtimes[0].completion is not runtimes[1].completion


@pytest.mark.parametrize("role", ["keyword", "embedding"])
def test_real_framework_queues_stop_before_close_returns(tmp_path, role):
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc, Tokenizer
    import numpy as np
    from tracemind.rag.qdrant import register_storage

    register_storage()

    class Tokens:
        def encode(self, text):
            return list(text.encode())

        def decode(self, tokens):
            return bytes(tokens).decode()

    async def exercise():
        entered, stopped = asyncio.Event(), asyncio.Event()

        async def blocked(*args, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async def embed(texts, **kwargs):
            if role == "embedding":
                await blocked()
            return np.zeros((len(texts), 2))

        rag = LightRAG(
            working_dir=str(tmp_path),
            workspace="queue-test-" + role,
            vector_storage="AsyncQdrantVectorDBStorage",
            vector_db_storage_cls_kwargs={
                "qdrant_url": "http://127.0.0.1:6333",
                "qdrant_timeout": 1,
            },
            tokenizer=Tokenizer("fixture", Tokens()),
            llm_model_func=blocked,
            embedding_func=EmbeddingFunc(
                embedding_dim=2, max_token_size=512, func=embed
            ),
        )
        runtime = RagRuntime(settings())
        runtime.rag = rag
        queue = (
            rag.role_llm_funcs[role]
            if role == "keyword"
            else rag.embedding_func.func
        )
        task = asyncio.create_task(
            queue("RPC timeout" if role == "keyword" else ["RPC timeout"])
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not stopped.is_set()
        await runtime.close()
        assert stopped.is_set()
        assert (await queue.get_queue_stats())["worker_count"] == 0
        await runtime.close()

    asyncio.run(exercise())


def test_close_drains_inference_even_when_cleanup_is_cancelled_twice():
    from lightrag.utils import priority_limit_async_func_call
    from tracemind.async_utils import run_io

    entered, release = threading.Event(), threading.Event()
    finalized = []

    def inference():
        entered.set()
        assert release.wait(3)

    async def exercise():
        async def embed():
            await run_io(inference)

        async def finalize():
            assert release.is_set()
            finalized.append(True)

        queue = priority_limit_async_func_call(1)(embed)
        runtime = RagRuntime(settings())
        runtime.rag = SimpleNamespace(
            role_llm_funcs={},
            embedding_func=SimpleNamespace(func=queue),
            finalize_storages=finalize,
        )
        task = asyncio.create_task(queue())
        await asyncio.to_thread(entered.wait)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        close = asyncio.create_task(runtime.close())
        await asyncio.sleep(0.01)
        close.cancel()
        await asyncio.sleep(0)
        close.cancel()
        await asyncio.sleep(0.01)
        assert not close.done() and not finalized
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await close
        assert finalized == [True]
        assert (await queue.get_queue_stats())["worker_count"] == 0

    try:
        asyncio.run(exercise())
    finally:
        release.set()


def test_same_snapshot_name_cannot_open_another_directory(
    tmp_path, monkeypatch
):
    import tracemind.rag.engine as engine

    monkeypatch.setattr(engine, "_snapshot_workspaces", {})
    opened = []

    async def initialize(runtime, **kwargs):
        opened.append(runtime.settings.workspace)

    monkeypatch.setattr(RagRuntime, "_initialize", initialize)

    async def exercise():
        first = RagRuntime(settings(directory=tmp_path / "a", snapshot="same"))
        await first.initialize()
        await first.close()
        second = RagRuntime(
            settings(directory=tmp_path / "b", snapshot="same")
        )
        with pytest.raises(ValueError, match="another local directory"):
            await second.initialize()
        assert second._workspace_lock is None
        assert len(opened) == 1

    asyncio.run(exercise())


def test_ambient_qdrant_workspace_cannot_override_snapshot(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("QDRANT_WORKSPACE", "some-other-snapshot")
    runtime = RagRuntime(settings(directory=tmp_path, snapshot="requested"))
    with pytest.raises(ValueError, match="QDRANT_WORKSPACE"):
        asyncio.run(runtime.initialize())
    assert not runtime.settings.workspace.exists()


@pytest.mark.parametrize("phase", ["initialize", "search", "close"])
@pytest.mark.parametrize("kind", ["connection", "service", "sqlite"])
def test_storage_failures_degrade_in_the_real_diagnosis_graph(
    monkeypatch, phase, kind
):
    import tracemind.rag.retriever as retriever
    from qdrant_client.http.exceptions import (
        ResponseHandlingException,
        UnexpectedResponse,
    )
    import httpx
    from test_online_knowledge import (
        END,
        INCIDENT,
        START,
        ScriptedModel,
        answer,
        service,
        tool_call,
    )

    failure = {
        "connection": ResponseHandlingException(httpx.ConnectError("private")),
        "service": UnexpectedResponse(
            503, "Unavailable", b"private", httpx.Headers()
        ),
        "sqlite": sqlite3.OperationalError("private"),
    }[kind]
    closed = []

    class Runtime:
        last_retrieval = {}

        def __init__(self, *args, **kwargs):
            pass

        async def initialize(self):
            if phase == "initialize":
                raise failure

        async def search(self, *args, **kwargs):
            if phase == "search":
                raise failure
            return []

        async def close(self):
            closed.append(True)
            if phase == "close":
                raise failure

    monkeypatch.setattr(retriever, "RagRuntime", Runtime)
    monkeypatch.setattr(
        retriever, "verify_snapshot", lambda path: {"snapshot": path.name}
    )
    search = SnapshotRetriever(settings())
    model = ScriptedModel(
        respond=lambda messages, calls: (
            tool_call("search_knowledge", {"query": "RPC timeout"})
            if calls == 1
            else answer(messages)
        )
    )
    result = asyncio.run(
        service(model, search).diagnose_with_evidence(
            INCIDENT, start=START, end=END
        )
    )
    assert result.state.status.value == "inconclusive"
    assert model.calls == 2
    assert result.state.knowledge == []
    assert closed == [True]
    assert "private" not in str(result.state.events)


def test_ir_metrics_include_zero_hits_and_deduplicate_documents():
    suite = RetrievalSuite(
        snapshot="v1",
        questions=[
            {
                "id": "one",
                "query": "RPC timeout",
                "relevant": {"synthetic.a": 1},
            },
            {
                "id": "two",
                "query": "Unknown error",
                "relevant": {"synthetic.b": 1},
            },
        ],
    )
    requested = []

    async def search(query):
        requested.append(query)
        if query.startswith("Unknown"):
            return []
        a = chunk("a")
        return [
            RetrievalHit(chunk=a, branches=["mix"], fusion_score=1),
            RetrievalHit(
                chunk=a.model_copy(update={"chunk_id": "a2"}),
                branches=["mix"],
                fusion_score=0.5,
            ),
        ]

    runtime = SimpleNamespace(
        settings=settings(snapshot="v1", result_limit=2),
        search=search,
        model=SimpleNamespace(
            calls=0, input_tokens=0, output_tokens=0, unreported_calls=0
        ),
    )
    report = asyncio.run(evaluate_retrieval(runtime, suite, k=2))
    assert requested == [q.query for q in suite.questions]
    assert report["queries"][0]["documents"] == ["synthetic.a"]
    assert set(report["metrics"].values()) == {0.5}
    with pytest.raises(ValueError, match="snapshot"):
        asyncio.run(
            evaluate_retrieval(
                runtime, suite.model_copy(update={"snapshot": "other"}), k=2
            )
        )


def test_abstention_questions_do_not_distort_positive_ranking_metrics():
    async def search(query):
        return (
            []
            if query == "unknown"
            else [
                RetrievalHit(
                    chunk=chunk("a"),
                    branches=["naive"],
                    fusion_score=1,
                    rerank_score=0.2,
                )
            ]
        )

    runtime = SimpleNamespace(
        settings=settings(snapshot="v1"),
        search=search,
        model=SimpleNamespace(
            calls=0, input_tokens=0, output_tokens=0, unreported_calls=0
        ),
    )
    suite = RetrievalSuite(
        snapshot="v1",
        questions=[
            {"id": "positive", "query": "RPC", "relevant": {"synthetic.a": 1}},
            {"id": "negative", "query": "unknown", "relevant": {}},
            {"id": "false-hit", "query": "wrong", "relevant": {}},
        ],
    )
    report = asyncio.run(evaluate_retrieval(runtime, suite, k=1))
    assert set(report["metrics"].values()) == {1}
    assert report["abstention"] == {
        "questions": 2,
        "correct": 1,
        "false_abstentions": 0,
    }
    assert report["queries"][0]["rerank_scores"] == [0.2]
    suite.questions = suite.questions[1:]
    report = asyncio.run(evaluate_retrieval(runtime, suite, k=1))
    assert report["metrics"] == {}


@pytest.mark.parametrize("score, expected", [(0.01, 0), (0.2, 1), (0.3, 1)])
def test_calibrated_threshold_can_return_no_knowledge(
    tmp_path, score, expected
):
    runtime = RagRuntime(settings(rerank_min_score=0.2))
    runtime._published = True
    runtime.index = KeywordIndex(tmp_path / "keyword.sqlite3")
    candidate = chunk("a")
    runtime.index.register([candidate])

    async def retrieve(*args):
        return {
            "status": "success",
            "data": {
                "chunks": [
                    {"chunk_id": candidate.chunk_id, "content": candidate.text}
                ]
            },
        }

    async def rank(*args):
        return [score]

    runtime.rag = SimpleNamespace(aquery_data=retrieve)
    runtime.models.rank = rank
    try:
        assert len(asyncio.run(runtime.search("RPC"))) == expected
    finally:
        runtime.index.close()
