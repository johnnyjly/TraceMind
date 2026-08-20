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
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
            return await PrometheusClient("http://prometheus", http).query(
                "up"
            )

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

    with pytest.raises(
        TelemetryClientError, match="Prometheus request failed"
    ):
        asyncio.run(run_query())


def test_metric_discovery_uses_window_and_escaped_service_matchers():
    service = 'payment"}or{job="other'
    start = datetime(2024, 9, 1, tzinfo=UTC)

    def handler(request):
        assert request.url.params.get_list("match[]") == [
            f'{{__name__=~".+",{label}={json.dumps(service)}}}'
            for label in ("service_name", "service", "job")
        ]
        assert request.url.params["start"] == start.isoformat()
        assert request.url.params["end"] == start.isoformat()
        assert request.url.params["limit"] == "3"
        if request.url.path == "/api/v1/label/__name__/values":
            return httpx.Response(200, json={"status": "success", "data": [
                "rpc_calls_total", "process_cpu_seconds_total",
            ]})
        assert request.url.path == "/api/v1/series"
        return httpx.Response(200, json={"status": "success", "data": [
            {"__name__": "rpc_calls_total", "job": service,
             "rpc_grpc_status_code": "13"},
            {"__name__": "rpc_calls_total", "job": service,
             "rpc_grpc_status_code": "0"},
            {"__name__": "process_cpu_seconds_total", "job": service},
        ]})

    async def discover():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        ) as http:
            return await PrometheusClient("http://prometheus", http)\
                .discover_metrics(service=service, start=start, end=start,
                                  limit=3)

    result = asyncio.run(discover())
    assert result["kind"] == "metric_metadata"
    assert result["possibly_truncated"]
    assert result["returned_series"] == 3
    rpc = next(m for m in result["metrics"]
               if m["metric"] == "rpc_calls_total")
    assert rpc["label_keys"] == ["job", "rpc_grpc_status_code"]
    assert {e["rpc_grpc_status_code"] for e in rpc["example_labels"]} == {
        "0", "13",
    }
    assert "id" not in result
    assert "without samples" in result["guidance"]


def test_metric_discovery_bounds_response_and_marks_omissions():
    # Simulate a backend ignoring its requested limit and large label sets.
    series = [{"__name__": f"metric_{i:03}", **{
        f"label_{j}": "v" * 150 for j in range(40)
    }} for i in range(100)]

    async def discover():
        transport = httpx.MockTransport(lambda request: httpx.Response(
            200, json={"status": "success", "data": (
                [item["__name__"] for item in series]
                if request.url.path.endswith("values") else series
            )},
        ))
        async with httpx.AsyncClient(transport=transport) as http:
            return await PrometheusClient("http://prometheus", http)\
                .discover_metrics(start=datetime(2024, 9, 1, tzinfo=UTC),
                                  end=datetime(2024, 9, 1, tzinfo=UTC),
                                  limit=50)

    result = asyncio.run(discover())
    assert len(json.dumps(result, ensure_ascii=False).encode()) <= 8192
    assert result["returned_series"] == 50
    assert result["possibly_truncated"]
    assert result["omitted_metrics"] > 0
    assert all(m["labels_truncated"] for m in result["metrics"])
    assert all(int(m["metric"].split("_")[1]) < 50
               for m in result["metrics"])


def test_name_catalogue_survives_histogram_series_limit_and_targets_labels():
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("values"):
            data = ["http_duration_bucket", "rpc_calls_total"]
        elif '"rpc_calls_total"' in request.url.params["match[]"]:
            data = [{"__name__": "rpc_calls_total", "job": "payment",
                     "rpc_grpc_status_code": "13"}]
        else:
            # All sampled series belong to the same metric, but the distinct
            # names endpoint still reveals another metric for investigation.
            data = [{"__name__": "http_duration_bucket", "le": str(i)}
                    for i in range(2)]
        return httpx.Response(200, json={"status": "success", "data": data})

    async def discover():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        ) as http:
            client = PrometheusClient("http://prometheus", http)
            now = datetime(2024, 9, 1, tzinfo=UTC)
            catalogue = await client.discover_metrics(
                service="payment", start=now, end=now, limit=2,
            )
            targeted = await client.discover_metrics(
                service="payment", metric="rpc_calls_total",
                start=now, end=now, limit=2,
            )
            return catalogue, targeted

    catalogue, targeted = asyncio.run(discover())
    rpc = next(m for m in catalogue["metrics"]
               if m["metric"] == "rpc_calls_total")
    assert rpc["example_labels"] == []
    assert catalogue["possibly_truncated"]
    assert targeted["metrics"][0]["example_labels"] == [
        {"job": "payment", "rpc_grpc_status_code": "13"},
    ]
    assert len(requests) == 3  # Targeted lookup does not reread the names.
    assert requests[-1].url.params.get_list("match[]") == [
        f'{{__name__="rpc_calls_total",{label}="payment"}}'
        for label in ("service_name", "service", "job")
    ]


@pytest.mark.parametrize("payload", [
    {"status": "error", "error": "invalid metadata request"},
    {"status": "success", "data": {}},
    {"status": "success", "data": [{"job": "payment"}]},
    {"status": "success", "data": [{"__name__": "metric", "job": 1}]},
])
def test_metric_discovery_rejects_invalid_backend_responses(payload):
    async def discover():
        transport = httpx.MockTransport(lambda request: httpx.Response(
            200, json=payload,
        ))
        async with httpx.AsyncClient(transport=transport) as http:
            return await PrometheusClient("http://prometheus", http)\
                .discover_metrics(start=datetime(2024, 9, 1, tzinfo=UTC),
                                  end=datetime(2024, 9, 1, tzinfo=UTC))

    with pytest.raises(TelemetryClientError, match="Prometheus"):
        asyncio.run(discover())


def test_jaeger_search_returns_trace_evidence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/traces"
        assert request.url.params["service"] == "checkout"
        assert request.url.params["start"] == "1725148800000000"
        assert request.url.params["end"] == "1725152400000000"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "traceID": "trace-123",
                        "spans": [
                            {
                                "startTime": 1_725_148_800_000_000,
                                "duration": 250_000,
                                "tags": [],
                            },
                            {
                                "startTime": 1_725_148_800_050_000,
                                "duration": 100_000,
                                "tags": [{"key": "error", "value": True}],
                            },
                        ],
                    }
                ]
            },
        )

    async def search_traces():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
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


def test_jaeger_search_scopes_mixed_age_spans_to_inclusive_window():
    """An old span must not date or inflate a current trace summary."""
    start = datetime(2024, 9, 1, tzinfo=UTC)
    end = datetime(2024, 9, 1, 1, tzinfo=UTC)
    first = int(start.timestamp() * 1_000_000)
    last = int(end.timestamp() * 1_000_000)

    def span(at, error=False):
        return {"startTime": at, "duration": 100_000,
                "tags": [{"key": "error", "value": error}]}

    payload = {"data": [
        {"traceID": "mixed", "spans": [
            span(first - 36_000_000_000, True), span(first),
            span(last, True), span(last + 1, True),
        ]},
        {"traceID": "old-only", "spans": [span(first - 1, True)]},
    ]}

    async def search():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload)
        )) as http:
            return await JaegerClient("http://jaeger", http).search_traces(
                "checkout", start=start, end=end,
            )

    evidence = asyncio.run(search())
    assert len(evidence) == 1
    assert evidence[0].observed_at == start
    attrs = evidence[0].attributes
    assert attrs["span_count"] == 2
    assert attrs["error_span_count"] == 1
    assert attrs["excluded_span_count"] == 2
    assert attrs["summary_scope"] == "spans_started_in_window"
    assert attrs["duration_ms"] == 3_600_100
    assert "selected span extent" in evidence[0].summary


def test_jaeger_trace_detail_returns_error_span_evidence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/traces/trace-123"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "traceID": "trace-123",
                        "processes": {
                            "p1": {"serviceName": "checkout"},
                            "p2": {"serviceName": "payment"},
                        },
                        "spans": [
                            {
                                "spanID": "span-ok",
                                "operationName": "PlaceOrder",
                                "processID": "p1",
                                "startTime": 1_725_192_000_000_000,
                                "duration": 300_000,
                                "tags": [],
                            },
                            {
                                "spanID": "span-error",
                                "operationName": "Charge",
                                "processID": "p2",
                                "startTime": 1_725_192_000_050_000,
                                "duration": 100_000,
                                "tags": [
                                    {
                                        "key": "otel.status_code",
                                        "value": "ERROR",
                                    },
                                    {
                                        "key": "error.type",
                                        "value": "PaymentFailureError",
                                    },
                                ],
                            },
                        ],
                    }
                ]
            },
        )

    async def get_detail():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
            return await JaegerClient("http://jaeger", http).get_trace_detail(
                "trace-123"
            )

    evidence = asyncio.run(get_detail())

    assert len(evidence) == 1
    assert evidence[0].source is EvidenceSource.TRACE
    assert evidence[0].service == "payment"
    assert evidence[0].summary == "Error span in payment: Charge"
    assert evidence[0].attributes["error_type"] == "PaymentFailureError"


def test_opensearch_query_returns_log_evidence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/otel-logs-*/_search"
        request_body = json.loads(request.content)
        filters = request_body["query"]["bool"]["filter"]
        assert filters[0] == {
            "term": {"resource.service.name.keyword": "payment"}
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
                                "severity": {"text": "ERROR", "number": 17},
                                "body": "Payment request timed out",
                                "traceId": "trace-123",
                                "spanId": "span-123",
                                "resource": {"service.name": "payment"},
                            },
                        }
                    ]
                }
            },
        )

    async def search_logs():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
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
    assert evidence[0].attributes["severity_text"] == "ERROR"
    assert evidence[0].attributes["severity_number"] == 17
    assert evidence[0].attributes["trace_id"] == "trace-123"
