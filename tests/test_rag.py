"""Validate RAG boundaries with synthetic references and no model downloads."""
# Optional dependencies are checked before retrieval imports.
# flake8: noqa: E402

import asyncio
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
pytest.importorskip("langchain_classic", reason="requires the rag extra")
pytest.importorskip("lightrag", reason="requires the rag extra")
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from tracemind.rag.config import RagSettings
from tracemind.rag.engine import RagRuntime
from tracemind.rag.keyword import KeywordIndex
from tracemind.rag.model_bridge import LightRAGModelBridge
from tracemind.rag.local_models import LocalModels
from tracemind.rag.fusion import fuse_candidates
from tracemind.rag.models import KnowledgeChunk

pytestmark = pytest.mark.rag


def chunk(key, text="Inspect connection errors.", **updates):
    """Create a synthetic source unrelated to Demo ground-truth labels."""
    return KnowledgeChunk(
        chunk_id=key,
        doc_id="synthetic." + key,
        title="Connections",
        section="Checks",
        text=text,
        document_path="synthetic.md",
        document_sha256="a" * 64,
        source_refs=["synthetic-source"],
        source_revision="test-v1",
        **updates,
    )


def settings(**updates):
    """Ignore private environment configuration in isolated tests."""
    return RagSettings(_env_file=None, **updates)


def test_example_config_and_invalid_bounds():
    configured = RagSettings(_env_file=Path(".env.example"))
    assert configured.embedding_dimension == 384
    assert configured.mode == "mix"
    for update in (
        {"snapshot": "../outside"},
        {"chunk_tokens": 510},
        {"result_limit": 21},
    ):
        with pytest.raises(ValidationError):
            settings(**update)


def test_bm25_stemming_literal_identifiers_replacement_and_hash(tmp_path):
    index = KeywordIndex(tmp_path / "index.sqlite3")
    try:
        first = chunk(
            "a",
            "A connection error needs checking.",
            identifiers="rpc.deadline_exceeded DB_POOL_LIMIT",
        )
        index.register([first, chunk("b", "Read the deployment guide.")])
        assert index.search("connections", limit=3)[0].chunk_id == "a"
        assert index.search("rpc.deadline_exceeded", limit=3)[0] == first
        assert index.search('" OR * --', limit=3) == []
        index.register(
            [
                first.model_copy(
                    update={
                        "text": "A replacement describes queues.",
                        "identifiers": "",
                    }
                )
            ]
        )
        assert index.count == 2
        assert index.search("rpc.deadline_exceeded", limit=3) == []
        assert index.search("error", limit=3) == []
        payload = json.loads(
            index._db.execute(
                "SELECT payload FROM chunks WHERE chunk_id='a'"
            ).fetchone()[0]
        )
        payload["text"] = "tampered"
        index._db.execute(
            "UPDATE chunks SET payload=? WHERE chunk_id='a'",
            (json.dumps(payload),),
        )
        with pytest.raises(ValueError, match="hash"):
            index.get("a")
        with pytest.raises(ValueError, match="provenance"):
            index.get("unknown")
    finally:
        index.close()


def test_fusion_deduplicates_by_id_and_rejects_conflicting_provenance():
    a, b = chunk("a"), chunk("b")
    hits = fuse_candidates({"graph": [a, a, b], "bm25": [a]}, limit=10)
    assert hits[0].chunk == a
    assert hits[0].branches == ["graph", "bm25"]
    with pytest.raises(ValueError, match="conflicting"):
        fuse_candidates(
            {
                "graph": [a],
                "bm25": [a.model_copy(update={"source_revision": "other"})],
            },
            limit=10,
        )


def test_local_embedding_prefix_and_token_limit(monkeypatch):
    models = LocalModels(settings(embedding_max_tokens=512))
    recorded = []

    class Embedding:
        def tokenizer(self, values, **kwargs):
            return {"input_ids": [[1] * len(value) for value in values]}

        def encode(self, values, **kwargs):
            recorded.extend(values)
            assert kwargs["normalize_embeddings"] is True
            return [[0] * 384 for _ in values]

    models.embedding = Embedding()
    monkeypatch.setattr(models, "load", lambda: None)

    async def exercise():
        await models.embed(["connection failure"], context="query")
        await models.embed(["reference text"], context="document")
        with pytest.raises(ValueError, match="tokenizer limit"):
            await models.embed(["x" * 513])

    asyncio.run(exercise())
    assert recorded == [
        models.settings.embedding_query_prefix + "connection failure",
        "reference text",
    ]


def test_internal_model_budget_usage_and_failed_attempts():
    class FakeModel:
        fail = False

        async def ainvoke(self, messages, **kwargs):
            assert kwargs["max_tokens"] == 2048
            assert kwargs["keyword_extraction"] is False
            if self.fail:
                raise OSError("injected failure")
            return AIMessage(
                content="{}",
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            )

    client = FakeModel()
    model = LightRAGModelBridge(settings(llm_max_calls=2), client.ainvoke)

    async def exercise():
        await model.complete("query")
        client.fail = True
        with pytest.raises(OSError):
            await model.complete("query")
        with pytest.raises(RuntimeError, match="budget"):
            await model.complete("query")

    asyncio.run(exercise())
    assert model.calls == 2
    assert (model.input_tokens, model.output_tokens) == (10, 5)
    assert model.unreported_calls == 1


def test_model_requires_injection_and_forwards_keyword_intent():
    missing = LightRAGModelBridge(settings())
    requests = []

    async def complete(messages, **kwargs):
        requests.append((messages, kwargs))
        return AIMessage(content="{}")

    model = LightRAGModelBridge(
        settings(llm_provider="external-provider"), complete)

    async def exercise():
        with pytest.raises(RuntimeError, match="injected"):
            await missing.complete("query")
        await model.complete("query", keyword_extraction=True)

    asyncio.run(exercise())
    assert missing.calls == 0
    assert requests[0][1]["keyword_extraction"] is True
    assert model.unreported_calls == 1


@pytest.mark.parametrize(
    "provider",
    [
        "google",
        "openai",
        "deepseek",
        "unsupported-provider",
    ],
)
def test_cli_chooses_provider_lazily_and_reuses_client(monkeypatch, provider):
    from tracemind import config
    from tracemind.agent.runtime import llm

    original_settings = config.Settings
    monkeypatch.setattr(
        config,
        "Settings",
        lambda **kwargs: original_settings(_env_file=None, **kwargs),
    )
    created, bindings, requests = [], [], []

    class FakeModel:
        def bind(self, **kwargs):
            bindings.append(kwargs)
            return self

        async def ainvoke(self, messages, **kwargs):
            requests.append(kwargs)
            return AIMessage(content="{}")

    def factory(configured):
        created.append((configured.llm_provider, configured.llm_model))
        return FakeModel()

    monkeypatch.setattr(llm, "create_chat_model", factory)
    module = runpy.run_path(str(Path("scripts/rag.py")))
    complete = module["create_completion"](
        settings(llm_provider=provider, llm_model="configured-model")
    )
    assert created == []

    async def exercise():
        for keyword in (True, False):
            await complete([], max_tokens=32, keyword_extraction=keyword)

    if provider == "unsupported-provider":
        with pytest.raises(ValidationError):
            asyncio.run(exercise())
        assert created == requests == bindings == []
        return
    asyncio.run(exercise())
    assert created == [(provider, "configured-model")]
    assert requests == [{"max_tokens": 32}, {"max_tokens": 32}]
    assert bindings == (
        []
        if provider == "google"
        else [{"response_format": {"type": "json_object"}}]
    )


@pytest.mark.parametrize("no_vector_results", [False, True])
def test_retrieval_fuses_bm25_reranks_and_rejects_unregistered_chunks(
    tmp_path,
    monkeypatch,
    no_vector_results,
):
    monkeypatch.setitem(
        sys.modules,
        "lightrag",
        SimpleNamespace(QueryParam=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    runtime = RagRuntime(settings(result_limit=2))
    runtime._published = True
    runtime.index = KeywordIndex(tmp_path / "keyword.sqlite3")
    a = chunk("a", "Connection failures need telemetry checks.")
    b = chunk("b", "A deployment guide.", identifiers="connection")
    runtime.index.register([a, b])

    class FakeRag:
        unknown = False

        async def aquery_data(self, query, param):
            assert param.only_need_context and not param.enable_rerank
            if self.unknown:
                return {
                    "status": "success",
                    "data": {
                        "chunks": [
                            {"chunk_id": "unknown", "content": "unregistered"}
                        ]
                    },
                }
            if no_vector_results:
                return {
                    "status": "failure",
                    "data": {},
                    "metadata": {"failure_reason": "no_results"},
                }
            return {
                "status": "success",
                "data": {
                    "chunks": [{"chunk_id": a.chunk_id, "content": a.text}]
                },
            }

        async def finalize_storages(self):
            pass

    async def rank(query, texts):
        return [2.0 if text == b.text else 1.0 for text in texts]

    runtime.rag = FakeRag()
    monkeypatch.setattr(runtime.models, "rank", rank)

    async def exercise():
        try:
            hits = await runtime.search("connection", mode="naive")
            assert [hit.chunk.chunk_id for hit in hits] == ["b", "a"]
            assert hits[0].branches == ["bm25"]
            assert runtime.model.calls == 0
            stages = runtime.last_retrieval["stages"]
            assert [hit["evaluation_id"]
                    for hit in stages["final"]] == [b.doc_id, a.doc_id]
            assert stages["final"][0]["document_sha256"] == b.document_sha256
            assert set(stages) == {"naive", "bm25", "fused", "final"}
            runtime.rag.unknown = True
            with pytest.raises(ValueError, match="provenance"):
                await runtime.search("connection")
        finally:
            await runtime.close()

    asyncio.run(exercise())
