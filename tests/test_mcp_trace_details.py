"""Ensure successful slow dependencies remain visible to MCP clients."""

import asyncio
from pathlib import Path
import sys

import httpx

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1] / "services/otel-telemetry-mcp/src"
    ),
)

from otel_telemetry_mcp.telemetry.jaeger import JaegerClient  # noqa: E402


def test_details_include_slow_successful_dependency_and_preserve_errors():
    trace_id = "a" * 32
    spans = [
        {
            "spanID": f"{i:016x}",
            "processID": "shipping",
            "operationName": "POST /ship-order",
            "startTime": 1725148800000000,
            "duration": (i + 1) * 1000,
            "tags": [],
            "references": [
                {
                    "refType": "CHILD_OF",
                    "traceID": trace_id,
                    "spanID": "b" * 16,
                }
            ],
        }
        for i in range(25)
    ]
    failed = {
        **spans[0],
        "spanID": "c" * 16,
        "tags": [{"key": "error", "value": True}],
    }
    requests = []

    def handler(request):
        """Return both an error and many successful spans in one trace."""
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "spans": [failed, *spans],
                        "processes": {"shipping": {"serviceName": "shipping"}},
                    }
                ]
            },
        )

    async def detail():
        """Exercise the independently packaged telemetry decoder."""
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        ) as http:
            return await JaegerClient("http://jaeger", http).get_trace_detail(
                trace_id
            )

    items = asyncio.run(detail())
    assert len(requests) == 1
    assert len(items) == 25
    assert items[0].summary == "Error span in shipping: POST /ship-order"
    assert items[0].attributes["is_error"] is True
    assert [e.attributes["duration_ms"] for e in items[1:]] == list(
        range(25, 1, -1)
    )
    assert all(not e.attributes["is_error"] for e in items[1:])
    assert items[1].service == "shipping"
    assert items[1].attributes["timestamp_source"] == "span_startTime"
    assert items[1].observed_at.timestamp() == 1725148800
    assert items[1].attributes["parent_span_id"] == "b" * 16


def test_filtered_pages_retain_fast_controls_and_diagnostic_fields():
    trace_id = "a" * 32
    spans = [
        {
            "spanID": str(i),
            "processID": "cart",
            "operationName": "GetCart",
            "startTime": 1725148800000000,
            "duration": i * 1000,
            "tags": [
                {"key": "user.id", "value": "example-user"},
                {"key": "server.address", "value": "valkey"},
                {"key": "authorization", "value": "never-export"},
            ],
        }
        for i in range(1, 8)
    ]

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "spans": spans,
                                "processes": {"cart": {"serviceName": "cart"}},
                            }
                        ]
                    },
                ),
            )
        ) as http:
            client = JaegerClient("http://jaeger", http)
            first = await client.get_trace_detail(
                trace_id, service="cart", limit=4
            )
            second = await client.get_trace_detail(
                trace_id, service="cart", offset=4, limit=4
            )
            assert {e.attributes["span_id"] for e in first + second} == {
                str(i) for i in range(1, 8)
            }
            assert first[0].attributes["coverage"]["next_offset"] == 4
            assert second[0].attributes["coverage"]["next_offset"] is None
            assert (
                first[0].attributes["coverage"]["trace_fingerprint"]
                == second[0].attributes["coverage"]["trace_fingerprint"]
            )
            diagnostic = first[0].attributes["diagnostic"]
            assert diagnostic["server.address"] == "valkey"
            assert "user.id.sha256" in diagnostic
            assert (
                "user.id" not in diagnostic
                and "authorization" not in diagnostic
            )
            empty = await client.get_trace_detail(
                trace_id, operation="EmptyCart"
            )
            assert empty["matched_span_count"] == 0

    asyncio.run(exercise())


def test_fingerprint_ignores_volatile_warnings_but_detects_span_changes():
    from copy import deepcopy

    trace_id = "a" * 32
    span = {"spanID": "1", "processID": "p", "operationName": "Charge",
            "startTime": 1725148800000000, "duration": 1000, "tags": []}
    payload = {"spans": [span], "processes": {"p": {"serviceName": "payment"}}}
    reads = []

    def handle(request):
        trace = deepcopy(payload)
        trace["spans"][0]["warnings"] = [f"volatile warning {len(reads)}"]
        if len(reads) == 2:
            trace["spans"][0]["duration"] += 1000
        reads.append(trace)
        return httpx.Response(200, json={"data": [trace]})

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle),
        ) as http:
            client = JaegerClient("http://jaeger", http)
            return [await client.get_trace_detail(trace_id) for _ in range(3)]

    results = asyncio.run(exercise())
    hashes = [rows[0].attributes["coverage"]["trace_fingerprint"]
              for rows in results]
    assert hashes[0] == hashes[1] and hashes[1] != hashes[2]
    assert results[0][0].id != results[1][0].id
