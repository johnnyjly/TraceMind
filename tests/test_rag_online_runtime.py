"""Check runtime ownership, batched ranking and independent IR evaluation."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from tracemind.evaluation.retrieval import RetrievalSuite, evaluate_retrieval
from tracemind.rag.config import RagSettings
from tracemind.rag.engine import RagRuntime
from tracemind.rag.keyword import KeywordIndex
from tracemind.rag.models import RetrievalHit
from tracemind.rag.retriever import SnapshotRetriever
from test_rag import chunk, settings


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
        return {"status": "success", "data": {"chunks": [
            {"chunk_id": item.chunk_id, "content": item.text}]}}

    async def rank(query, texts):
        ranks.append((query, texts))
        return [3 if text == b.text else 1 for text in texts]

    async def close():
        pass

    runtime.rag = SimpleNamespace(aquery_data=retrieve,
                                  finalize_storages=close)
    runtime.models.rank = rank

    async def exercise():
        try:
            hits = await runtime.search("slow dependency", queries=[
                "RPC timeout", "Queue saturation"])
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
    monkeypatch.setattr(retriever, "verify_snapshot",
                        lambda path: {"snapshot": path.name})
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
            search(["two"], question="two", snapshot="v1", completion=second))
        task = asyncio.create_task(search(["three"], question="cancel",
                                          snapshot="v1", completion=first))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert [
        m.content for m in received] == [
        "first-run",
        "second-run",
        "first-run"]
    assert all(r.closed for r in runtimes)
    assert len({id(r.models) for r in runtimes}) == 1
    assert runtimes[0].completion is not runtimes[1].completion


def test_ir_metrics_include_zero_hits_and_deduplicate_documents():
    suite = RetrievalSuite(snapshot="v1", questions=[
        {"id": "one", "query": "RPC timeout", "relevant": {"synthetic.a": 1}},
        {"id": "two", "query": "Unknown error",
         "relevant": {"synthetic.b": 1}},
    ])
    requested = []

    async def search(query):
        requested.append(query)
        if query.startswith("Unknown"):
            return []
        a = chunk("a")
        return [RetrievalHit(chunk=a, branches=["mix"], fusion_score=1),
                RetrievalHit(chunk=a.model_copy(update={"chunk_id": "a2"}),
                             branches=["mix"], fusion_score=.5)]

    runtime = SimpleNamespace(
        settings=settings(snapshot="v1", result_limit=2), search=search,
        model=SimpleNamespace(calls=0, input_tokens=0, output_tokens=0,
                              unreported_calls=0))
    report = asyncio.run(evaluate_retrieval(runtime, suite, k=2))
    assert requested == [q.query for q in suite.questions]
    assert report["queries"][0]["documents"] == ["synthetic.a"]
    assert set(report["metrics"].values()) == {.5}
    with pytest.raises(ValueError, match="snapshot"):
        asyncio.run(evaluate_retrieval(
            runtime, suite.model_copy(update={"snapshot": "other"}), k=2))
