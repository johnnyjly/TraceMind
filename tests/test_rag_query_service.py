"""Service lifecycle, isolation, overload and counted remote keywords."""

import asyncio
from types import SimpleNamespace

import httpx
from langchain_core.messages import AIMessage
import pytest

from tracemind.rag.config import RagSettings
from tracemind.rag.query_client import HttpSnapshotRetriever
from tracemind.rag.query_service import create_app


class Runtime:
    last_retrieval = {}

    def retrieval_policy(self):
        return {"mode": "mix"}

    def __init__(self):
        self.initializations = self.closed = 0
        self.embeddings = self.rankings = 0
        self.calls = []
        self.models = self
        self.block = None

    async def initialize(self):
        self.initializations += 1

    async def close(self):
        self.closed += 1

    async def embed(self, *args, **kwargs):
        self.embeddings += 1

    async def rank(self, *args):
        self.rankings += 1

    async def search(self, question, **kwargs):
        self.calls.append((question, kwargs))
        if self.block:
            await self.block.wait()
        return []


def payload(snapshot="test"):
    return {
        "snapshot": snapshot,
        "mode": "mix",
        "question": "payment error",
        "queries": [
            {
                "query": "payment error",
                "high_level_keywords": ["failure"],
                "low_level_keywords": ["payment"],
            }
        ],
    }


def app_for(monkeypatch, runtime, **settings):
    monkeypatch.setattr(
        "tracemind.rag.query_service.verify_snapshot",
        lambda _: {"snapshot": "test"},
    )
    return create_app(
        RagSettings(_env_file=None, snapshot="test", **settings),
        runtime=runtime,
    )


def test_service_prewarms_once_reuses_owner_and_rejects_wrong_snapshot(
    monkeypatch,
):
    async def exercise():
        runtime = Runtime()
        app = app_for(monkeypatch, runtime)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://rag"
        ) as c:
            assert (await c.get("/ready")).status_code == 503
            async with app.router.lifespan_context(app):
                assert (await c.get("/ready")).json()["snapshot"] == "test"
                assert (
                    await c.post("/search", json=payload("other"))
                ).status_code == 409
                for _ in range(2):
                    assert (
                        await c.post("/search", json=payload())
                    ).status_code == 200
                invalid = payload()
                invalid["queries"][0]["high_level_keywords"] = []
                invalid["queries"][0]["low_level_keywords"] = []
                assert (
                    await c.post("/search", json=invalid)
                ).status_code == 422
                assert runtime.initializations == runtime.embeddings == 1
                assert runtime.rankings == 1
                assert len(runtime.calls) == 2
                assert runtime.calls[0][1]["query_keywords"] == [
                    (["failure"], ["payment"])
                ]
            assert runtime.closed == 1
            assert (await c.get("/ready")).status_code == 503

    asyncio.run(exercise())


def test_service_rejects_overload_and_releases_timeout(monkeypatch):
    async def exercise():
        runtime = Runtime()
        runtime.block = asyncio.Event()
        app = app_for(monkeypatch, runtime, service_query_timeout=0.05)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://rag",
            ) as c:
                first = asyncio.create_task(c.post("/search", json=payload()))
                while not runtime.calls:
                    await asyncio.sleep(0)
                assert (
                    await c.post("/search", json=payload())
                ).status_code == 503
                assert (await first).status_code == 504
                runtime.block.set()
                assert (
                    await c.post("/search", json=payload())
                ).status_code == 200

    asyncio.run(exercise())


def test_http_retriever_forwards_counted_keywords(monkeypatch):
    async def exercise():
        requests, completions = [], []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"snapshot": "test", "hits": []})

        original = httpx.AsyncClient
        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda **kw: original(
                **kw, transport=httpx.MockTransport(handler)
            ),
        )

        async def complete(messages, **kwargs):
            completions.append((messages, kwargs))
            return AIMessage(
                content='{"high_level_keywords":["failure"],'
                '"low_level_keywords":["payment"]}'
            )

        search = HttpSnapshotRetriever(
            RagSettings(
                _env_file=None, service_url="http://rag", snapshot="test"
            )
        )
        assert (
            await search(
                ["payment error"],
                question="payment error",
                snapshot="test",
                completion=complete,
            )
        ).hits == []
        assert len(completions) == 1
        assert completions[0][1]["keyword_extraction"] is True
        import json

        body = json.loads(requests[-1].content)
        assert body == payload()
        assert "authorization" not in requests[-1].headers

    asyncio.run(exercise())


def test_startup_failure_closes_owner_without_readiness(monkeypatch):
    async def exercise():
        runtime = Runtime()

        async def fail(*args, **kwargs):
            raise RuntimeError("cannot warm model")

        runtime.models = SimpleNamespace(embed=fail)
        app = app_for(monkeypatch, runtime)
        with pytest.raises(RuntimeError, match="cannot warm"):
            async with app.router.lifespan_context(app):
                pytest.fail("must not become ready")
        assert runtime.closed == 1

    asyncio.run(exercise())


def test_naive_mode_and_empty_cause_reach_http_client(monkeypatch):
    async def exercise():
        runtime = Runtime()
        runtime.last_retrieval = {
            "empty_reason": "below_cutoff",
            "rejected_count": 3,
        }
        app = app_for(monkeypatch, runtime)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://rag"
            ) as client:
                request = {**payload(), "mode": "naive"}
                result = await client.post("/search", json=request)
                assert result.status_code == 200
                assert runtime.calls[0][1]["mode"] == "naive"
                assert (
                    result.json()["diagnostics"]["empty_reason"]
                    == "below_cutoff"
                )
                assert result.json()["diagnostics"]["rejected_count"] == 3

    asyncio.run(exercise())


def test_trace_context_is_request_scoped_and_diagnostics_are_compact(
    monkeypatch,
):
    from contextlib import contextmanager

    captures = []

    @contextmanager
    def observe(client, name, **kwargs):
        record = {"name": name, **kwargs}
        captures.append(record)
        yield SimpleNamespace(update=lambda **values: record.update(values))

    monkeypatch.setattr("tracemind.rag.query_service.observation", observe)

    async def exercise():
        runtime = Runtime()
        runtime.last_retrieval = {
            "query": "repeated text",
            "stages": {"raw": []},
            "returned_count": 0,
        }
        app = app_for(monkeypatch, runtime)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://rag"
            ) as client:
                await client.get("/ready")
                assert captures == []
                tid, pid = "a" * 32, "b" * 16
                good = await client.post(
                    "/search",
                    json=payload(),
                    headers={"traceparent": f"00-{tid}-{pid}-01"},
                )
                assert captures[-1]["trace_context"] == {
                    "trace_id": tid,
                    "parent_span_id": pid,
                }
                assert captures[-1]["metadata"] == {"http_status": 200}
                assert set(good.json()["diagnostics"]) == {
                    "returned_count",
                    "snapshot_fingerprint",
                }
                for header in [
                    "invalid",
                    "00-" + "0" * 32 + "-" + pid + "-01",
                    "",
                ]:
                    await client.post(
                        "/search",
                        json=payload("wrong"),
                        headers={"traceparent": header},
                    )
                    assert captures[-1]["trace_context"] is None
                    assert captures[-1]["level"] == "ERROR"
                    assert captures[-1]["metadata"] == {"http_status": 409}

    asyncio.run(exercise())


def test_http_retriever_propagates_current_observation(monkeypatch):
    async def exercise():
        requests = []

        def handle(request):
            requests.append(request)
            return httpx.Response(200, json={"snapshot": "test", "hits": []})

        original = httpx.AsyncClient
        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda **kw: original(**kw, transport=httpx.MockTransport(handle)),
        )
        client = SimpleNamespace(
            get_current_trace_id=lambda: "a" * 32,
            get_current_observation_id=lambda: "b" * 16,
        )
        search = HttpSnapshotRetriever(
            RagSettings(
                _env_file=None,
                service_url="http://rag",
                snapshot="test",
                mode="naive",
            ),
            langfuse_client=client,
        )
        await search(
            ["payment"], question="payment", snapshot="test", completion=None
        )
        assert "traceparent" not in requests[0].headers
        assert (
            requests[-1].headers["traceparent"]
            == "00-" + "a" * 32 + "-" + "b" * 16 + "-01"
        )

    asyncio.run(exercise())
