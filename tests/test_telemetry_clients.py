import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest

from tracemind.models import EvidenceSource
from tracemind.telemetry import (
    JaegerClient,
    OpenSearchClient,
    PrometheusClient,
    TelemetryClientError,
)


def test_prometheus_query_returns_metric_evidence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/query"
        assert request.url.params["query"] == "up"
        return httpx.Response(
            200,
            json={
                "status": "success",
                "data": {
                    "resultType": "vector",
                    "result": [
                        {
                            "metric": {
                                "__name__": "up",
                                "service_name": "checkout",
                            },
                            "value": [1_725_192_000, "1"],
                        }
                    ],
                },
            },
        )

    async def run_query():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await PrometheusClient("http://prometheus", http).query("up")

    evidence = asyncio.run(run_query())

    assert len(evidence) == 1
    assert evidence[0].source is EvidenceSource.METRIC
    assert evidence[0].service == "checkout"
    assert evidence[0].attributes["value"] == "1"


def test_prometheus_http_failure_has_backend_context() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(503))

    async def run_query() -> None:
        async with httpx.AsyncClient(transport=transport) as http:
            await PrometheusClient("http://prometheus", http).query("up")

    with pytest.raises(TelemetryClientError, match="Prometheus request failed"):
        asyncio.run(run_query())


def test_jaeger_search_returns_trace_evidence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/traces"
        assert request.url.params["service"] == "checkout"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "traceID": "trace-123",
                        "spans": [
                            {
                                "startTime": 1_725_192_000_000_000,
                                "duration": 250_000,
                                "tags": [],
                            },
                            {
                                "startTime": 1_725_192_000_050_000,
                                "duration": 100_000,
                                "tags": [{"key": "error", "value": True}],
                            },
                        ],
                    }
                ]
            },
        )

    async def search_traces():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await JaegerClient("http://jaeger", http).search_traces(
                "checkout",
                start=datetime(2024, 9, 1, tzinfo=UTC),
                end=datetime(2024, 9, 1, 1, tzinfo=UTC),
            )

    evidence = asyncio.run(search_traces())

    assert len(evidence) == 1
    assert evidence[0].source is EvidenceSource.TRACE
    assert evidence[0].attributes["trace_id"] == "trace-123"
    assert evidence[0].attributes["error_span_count"] == 1
    assert evidence[0].attributes["duration_ms"] == 250.0


def test_opensearch_query_returns_log_evidence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/otel-logs-*/_search"
        request_body = json.loads(request.content)
        filters = request_body["query"]["bool"]["filter"]
        assert filters[0] == {
            "term": {"resource.attributes.service.name": "payment"}
        }
        return httpx.Response(
            200,
            json={
                "hits": {
                    "hits": [
                        {
                            "_id": "log-123",
                            "_source": {
                                "@timestamp": "2024-09-01T00:30:00Z",
                                "severity_text": "ERROR",
                                "severity_number": 17,
                                "body": "Payment request timed out",
                                "traceId": "trace-123",
                                "spanId": "span-123",
                                "resource.attributes.service.name": "payment",
                            },
                        }
                    ]
                }
            },
        )

    async def search_logs():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await OpenSearchClient(
                "http://opensearch",
                "otel-logs-*",
                http,
            ).search_logs(
                "payment",
                start=datetime(2024, 9, 1, tzinfo=UTC),
                end=datetime(2024, 9, 1, 1, tzinfo=UTC),
                text="timed out",
            )

    evidence = asyncio.run(search_logs())

    assert len(evidence) == 1
    assert evidence[0].source is EvidenceSource.LOG
    assert evidence[0].summary == "Payment request timed out"
    assert evidence[0].attributes["trace_id"] == "trace-123"

